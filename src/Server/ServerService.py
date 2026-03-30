import signal
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import grpc
import asyncio
import logging
from concurrent import futures
from datetime import datetime

from generated import Server_pb2
from generated import Server_pb2_grpc
from grpc_reflection.v1alpha import reflection
from Server import ZarrService as ZarrService

logger = logging.getLogger(__name__)

zarr_service = ZarrService.ZarrService()


def _build_volume_metadata(meta: dict) -> Server_pb2.VolumeMetadata:
    return Server_pb2.VolumeMetadata(
        volume_id  = meta["volume_id"],
        shape      = meta["shape"],
        dtype      = meta["dtype"],
        size_bytes = meta["size_bytes"],
        created_at = meta["created_at"],
        chunkshape = meta.get("chunkshape", []),
        metadata   = meta["metadata"],
    )


def _build_mesh_metadata(meta: dict) -> Server_pb2.MeshMetadata:
    return Server_pb2.MeshMetadata(
        mesh_id             = meta["mesh_id"],
        vertices_shape      = meta["vertices_shape"],
        faces_shape         = meta["faces_shape"],
        vertices_dtype      = meta["vertices_dtype"],
        faces_dtype         = meta["faces_dtype"],
        vertices_size_bytes = meta["vertices_size_bytes"],
        faces_size_bytes    = meta["faces_size_bytes"],
        created_at          = meta["created_at"],
        vertices_chunkshape = meta.get("vertices_chunkshape", []),
        faces_chunkshape    = meta.get("faces_chunkshape", []),
        metadata            = meta["metadata"],
    )


class DataIngestServicer(Server_pb2_grpc.DataIngestServicer):

    def __init__(self):
        # global notification subscribers (SubscribeNotifications RPC)
        self._subscribers: list[asyncio.Queue] = []

        # per-id live stream queues for in-progress ingests
        # FetchVolume/FetchMesh register here while an ingest is running
        self._volume_stream_subs: dict[str, list[asyncio.Queue]] = {}
        self._mesh_stream_subs:   dict[str, list[asyncio.Queue]] = {}

    async def _notify_subscribers(self, notification: Server_pb2.Notification) -> None:
        for queue in self._subscribers:
            await queue.put(notification)

    async def _push_volume_chunk(self, volume_id: str, chunk: Server_pb2.Volume) -> None:
        for queue in self._volume_stream_subs.get(volume_id, []):
            await queue.put(chunk)

    async def _push_mesh_chunk(self, mesh_id: str, chunk: Server_pb2.Mesh) -> None:
        for queue in self._mesh_stream_subs.get(mesh_id, []):
            await queue.put(chunk)

    async def _close_volume_stream(self, volume_id: str) -> None:
        for queue in self._volume_stream_subs.get(volume_id, []):
            await queue.put(None)   # None is the end-of-stream sentinel
        self._volume_stream_subs.pop(volume_id, None)

    async def _close_mesh_stream(self, mesh_id: str) -> None:
        for queue in self._mesh_stream_subs.get(mesh_id, []):
            await queue.put(None)
        self._mesh_stream_subs.pop(mesh_id, None)

    async def IngestVolume(self, request_iterator, context):
        header = None

        async for chunk in request_iterator:
            if chunk.HasField("header"):
                header = chunk.header

                # reserve zarr space as soon as shape/dtype/chunks are known
                await zarr_service.prepare_volume(
                    volume_id = header.volume_id,
                    shape     = list(header.shape),
                    dtype     = header.dtype,
                    chunks    = list(header.chunkshape),
                    metadata  = dict(header.metadata),
                )

                # forward header to any live-stream subscribers immediately
                await self._push_volume_chunk(
                    header.volume_id,
                    Server_pb2.Volume(header=header),
                )
                logger.info("Pushing header for volume '%s' to live subscribers", header.volume_id)

            elif chunk.HasField("data"):
                if header is None:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("Data chunk received before header")
                    logger.error("Data chunk received before header")
                    return Server_pb2.IngestVolumeResponse(success=False, mesg="No header")

                # write to zarr and forward to live subscribers in parallel
                await asyncio.gather(
                    zarr_service.stream_volume_chunk(
                        volume_id  = header.volume_id,
                        chunk_data = chunk.data.data,
                    ),
                    self._push_volume_chunk(
                        header.volume_id,
                        Server_pb2.Volume(data=chunk.data),
                    ),
                )

        if header is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("No header received in stream")
            logger.error("No header received in stream")
            return Server_pb2.IngestVolumeResponse(success=False, mesg="No header")

        meta = await zarr_service.finalise_volume(
            volume_id  = header.volume_id,
            created_at = datetime.now().isoformat(),
            metadata   = dict(header.metadata),
        )

        if meta is None:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to finalise volume in zarr")
            logger.error("Failed to finalise volume '%s' in zarr", header.volume_id)
            return Server_pb2.IngestVolumeResponse(success=False, mesg="Finalise failed")

        await self._close_volume_stream(header.volume_id)

        await self._notify_subscribers(Server_pb2.Notification(
            type            = Server_pb2.ADDED,
            data_type       = Server_pb2.VOLUME,
            id              = meta["volume_id"],
            volume_metadata = _build_volume_metadata(meta),
        ))

        logger.info("Volume '%s' ingested", meta["volume_id"])
        return Server_pb2.IngestVolumeResponse(
            success   = True,
            mesg      = "Volume ingested successfully",
            volume_id = meta["volume_id"],
        )

    async def IngestMesh(self, request_iterator, context):
        header = None

        async for chunk in request_iterator:
            if chunk.HasField("header"):
                header = chunk.header

                await zarr_service.prepare_mesh(
                    mesh_id         = header.mesh_id,
                    vertices_shape  = list(header.vertices_shape),
                    faces_shape     = list(header.faces_shape),
                    vertices_dtype  = header.vertices_dtype,
                    faces_dtype     = header.faces_dtype,
                    vertices_chunks = list(header.vertices_chunkshape),
                    faces_chunks    = list(header.faces_chunkshape),
                    metadata        = dict(header.metadata),
                )

                await self._push_mesh_chunk(
                    header.mesh_id,
                    Server_pb2.Mesh(header=header),
                )
                logger.info("Pushing header for mesh '%s' to live subscribers", header.mesh_id)

            elif chunk.HasField("data"):
                if header is None:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details("Data chunk received before header")
                    logger.error("Data chunk received before header")
                    return Server_pb2.IngestMeshResponse(success=False, mesg="No header")

                array_name = "vertices" if chunk.data.HasField("vertices") else "faces"
                chunk_data = chunk.data.vertices if array_name == "vertices" else chunk.data.faces

                await asyncio.gather(
                    zarr_service.stream_mesh_chunk(
                        mesh_id    = header.mesh_id,
                        array_name = array_name,
                        chunk_data = chunk_data,
                    ),
                    self._push_mesh_chunk(
                        header.mesh_id,
                        Server_pb2.Mesh(data=chunk.data),
                    ),
                )

        if header is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("No header received in stream")
            logger.error("No header received in stream")
            return Server_pb2.IngestMeshResponse(success=False, mesg="No header")

        meta = await zarr_service.finalise_mesh(
            mesh_id    = header.mesh_id,
            created_at = datetime.now().isoformat(),
            metadata   = dict(header.metadata),
        )

        if meta is None:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to finalise mesh in zarr")
            logger.error("Failed to finalise mesh '%s' in zarr", header.mesh_id)
            return Server_pb2.IngestMeshResponse(success=False, mesg="Finalise failed")

        await self._close_mesh_stream(header.mesh_id)

        await self._notify_subscribers(Server_pb2.Notification(
            type          = Server_pb2.ADDED,
            data_type     = Server_pb2.MESH,
            id            = meta["mesh_id"],
            mesh_metadata = _build_mesh_metadata(meta),
        ))

        logger.info("Mesh '%s' ingested", meta["mesh_id"])
        return Server_pb2.IngestMeshResponse(
            success = True,
            mesg    = "Mesh ingested successfully",
            mesh_id = meta["mesh_id"],
        )


    def ListData(self, request, context):
        response = Server_pb2.DataList()
        for meta in zarr_service.list_volumes():
            response.volumes.append(_build_volume_metadata(meta))
        for meta in zarr_service.list_meshes():
            response.meshes.append(_build_mesh_metadata(meta))
        logger.info("Listed data: %d volumes, %d meshes", len(response.volumes), len(response.meshes))
        return response

    def GetVolumeMetatdata(self, request, context):
        meta = zarr_service.get_volume_metadata(request.id)
        if meta is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Volume '{request.id}' not found")
            logger.warning("Volume '%s' not found for metadata request", request.id)
            return Server_pb2.VolumeMetadata()
        logger.info("Retrieved metadata for volume '%s'", request.id)
        return _build_volume_metadata(meta)

    def GetMeshMetatdata(self, request, context):
        meta = zarr_service.get_mesh_metadata(request.id)
        if meta is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Mesh '{request.id}' not found")
            logger.warning("Mesh '%s' not found for metadata request", request.id)
            return Server_pb2.MeshMetadata()
        logger.info("Retrieved metadata for mesh '%s'", request.id)
        return _build_mesh_metadata(meta)

    # ------------------------------------------------------------------
    # Fetch RPCs
    # Two cases per type:
    #   A) already in cold storage  -> stream from zarr
    #   B) ingest in progress       -> subscribe to live queue, yield as chunks arrive
    # ------------------------------------------------------------------

    async def FetchVolume(self, request, context):
        volume_id = request.id

        # Case A: cold storage
        if zarr_service.get_volume_metadata(volume_id) is not None:
            logger.info("Fetching volume '%s' from cold storage", volume_id)
            meta = zarr_service.get_volume_metadata(volume_id)
            assert meta is not None

            yield Server_pb2.Volume(header=Server_pb2.VolumeHeader(
                volume_id  = meta["volume_id"],
                shape      = meta["shape"],
                dtype      = meta["dtype"],
                chunkshape = meta.get("chunkshape", []),
                metadata   = meta["metadata"],
            ))

            async for chunk_arr in zarr_service.read_volume_chunks(volume_id):
                yield Server_pb2.Volume(
                    data=Server_pb2.VolumeData(data=chunk_arr.tobytes())
                )
            return

        # Case B: ingest in progress — tap the live stream
        if volume_id in zarr_service._volume_pending:
            logger.info("Fetching volume '%s' from live stream (ingest in progress)", volume_id)
            queue: asyncio.Queue = asyncio.Queue(maxsize=10)
            self._volume_stream_subs.setdefault(volume_id, []).append(queue)
            try:
                while True:
                    item = await queue.get()
                    if item is None:    # end-of-stream sentinel
                        break
                    yield item
            finally:
                subs = self._volume_stream_subs.get(volume_id, [])
                if queue in subs:
                    subs.remove(queue)
            return

        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(f"Volume '{volume_id}' not found")
        logger.warning("Volume '%s' not found for fetch request", volume_id)

    async def FetchMesh(self, request, context):
        mesh_id = request.id

        # Case A: cold storage
        if zarr_service.get_mesh_metadata(mesh_id) is not None:
            logger.info("Fetching mesh '%s' from cold storage", mesh_id)
            meta = zarr_service.get_mesh_metadata(mesh_id)
            assert meta is not None

            yield Server_pb2.Mesh(header=Server_pb2.MeshHeader(
                mesh_id             = meta["mesh_id"],
                vertices_shape      = meta["vertices_shape"],
                faces_shape         = meta["faces_shape"],
                vertices_dtype      = meta["vertices_dtype"],
                faces_dtype         = meta["faces_dtype"],
                vertices_chunkshape = meta.get("vertices_chunkshape", []),
                faces_chunkshape    = meta.get("faces_chunkshape", []),
                metadata            = meta["metadata"],
            ))

            async for array_name, chunk_arr in zarr_service.read_mesh_chunks(mesh_id):
                if array_name == "vertices":
                    yield Server_pb2.Mesh(
                        data=Server_pb2.MeshData(vertices=chunk_arr.tobytes())
                    )
                else:
                    yield Server_pb2.Mesh(
                        data=Server_pb2.MeshData(faces=chunk_arr.tobytes())
                    )
            return

        # Case B: ingest in progress
        if mesh_id in zarr_service._mesh_pending:
            logger.info("Fetching mesh '%s' from live stream (ingest in progress)", mesh_id)
            queue: asyncio.Queue = asyncio.Queue()
            self._mesh_stream_subs.setdefault(mesh_id, []).append(queue)
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item
            finally:
                subs = self._mesh_stream_subs.get(mesh_id, [])
                if queue in subs:
                    subs.remove(queue)
            return

        context.set_code(grpc.StatusCode.NOT_FOUND)
        context.set_details(f"Mesh '{mesh_id}' not found")
        logger.warning("Mesh '%s' not found for fetch request", mesh_id)

    # ------------------------------------------------------------------
    # Notification subscription
    # ------------------------------------------------------------------

    async def SubscribeNotifications(self, request, context):
        logger.info("Client subscribed to notifications")
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        try:
            while True:
                notification = await queue.get()
                logger.info(
                    "Sending notification  type=%s  data_type=%s  id=%s",
                    Server_pb2.NotificationType.Name(notification.type),
                    Server_pb2.DataType.Name(notification.data_type),
                    notification.id,
                )
                yield notification
        finally:
            self._subscribers.remove(queue)
            logger.info("Client unsubscribed from notifications")


async def serve():
    await zarr_service.initialise()

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        compression=grpc.Compression.Gzip,
        options=[
            ("grpc.max_send_message_length",    100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ],
    )

    Server_pb2_grpc.add_DataIngestServicer_to_server(DataIngestServicer(), server)

    SERVICE_NAMES = (
        Server_pb2.DESCRIPTOR.services_by_name["DataIngest"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    server.add_insecure_port("[::]:50051")

    await server.start()
    logger.info("Server started on port 50051")

    # use an event instead of wait_for_termination so we control shutdown
    shutdown_event = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    # block here until signal arrives
    await shutdown_event.wait()

    logger.info("Shutdown signal received — stopping server ...")

    # grace period: finish in-flight RPCs before closing (5s)
    await server.stop(grace=5)

    logger.info("Server stopped cleanly")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())