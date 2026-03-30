import grpc
import threading
import itertools
import logging
import json
import math
import numpy as np
import nibabel as nib
from typing import Iterator
from generated import Server_pb2, Server_pb2_grpc
from medvol import MedVol

logger = logging.getLogger(__name__)
logging.getLogger("Server.ZarrService").setLevel(logging.DEBUG)
logging.getLogger(__name__).setLevel(logging.DEBUG)
class DataClient:

    def __init__(self, address: str = "localhost:50051"):
        self._address = address
        self._channel = grpc.insecure_channel(
            address,
            compression=grpc.Compression.Gzip,
        )
        self._stub    = Server_pb2_grpc.DataIngestStub(self._channel)
        self._pending: list[threading.Thread] = []
        logger.info("DataClient connected to %s", address)

    # ------------------------------------------------------------------
    # Internal — sync iterators, run in background threads
    # ------------------------------------------------------------------

    def _volume_iterator(
        self,
        volume_id: str,
        array:     np.ndarray,
        chunks:    tuple[int, ...],
        metadata:  dict | None = None,
    ) -> Iterator[Server_pb2.Volume]:
        yield Server_pb2.Volume(header=Server_pb2.VolumeHeader(
            volume_id  = volume_id,
            shape      = list(array.shape),
            dtype      = str(array.dtype),
            chunkshape = list(chunks),
        ))

        # ceil division — captures the last partial block
        n_blocks = [math.ceil(s / c) for s, c in zip(array.shape, chunks)]
        for block_idx in itertools.product(*[range(n) for n in n_blocks]):
            slices = tuple(
                slice(b * c, min((b + 1) * c, s))
                for b, c, s in zip(block_idx, chunks, array.shape)
            )
            block = np.ascontiguousarray(array[slices], dtype=array.dtype)
            yield Server_pb2.Volume(
                data=Server_pb2.VolumeData(data=block.tobytes(order='C'))
            )

    def _mesh_iterator(
        self,
        mesh_id:  str,
        vertices: np.ndarray,
        faces:    np.ndarray,
        v_chunks: tuple[int, ...],
        f_chunks: tuple[int, ...],
    ) -> Iterator[Server_pb2.Mesh]:
        yield Server_pb2.Mesh(header=Server_pb2.MeshHeader(
            mesh_id             = mesh_id,
            vertices_shape      = list(vertices.shape),
            faces_shape         = list(faces.shape),
            vertices_dtype      = str(vertices.dtype),
            faces_dtype         = str(faces.dtype),
            vertices_chunkshape = list(v_chunks),
            faces_chunkshape    = list(f_chunks),
        ))

        n_v_blocks = [s // c for s, c in zip(vertices.shape, v_chunks)]
        for block_idx in itertools.product(*[range(n) for n in n_v_blocks]):
            slices = tuple(
                slice(b * c, min((b + 1) * c, s))
                for b, c, s in zip(block_idx, v_chunks, vertices.shape)
            )
            yield Server_pb2.Mesh(
                data=Server_pb2.MeshData(
                    vertices=np.ascontiguousarray(vertices[slices]).tobytes()
                )
            )

        n_f_blocks = [s // c for s, c in zip(faces.shape, f_chunks)]
        for block_idx in itertools.product(*[range(n) for n in n_f_blocks]):
            slices = tuple(
                slice(b * c, min((b + 1) * c, s))
                for b, c, s in zip(block_idx, f_chunks, faces.shape)
            )
            yield Server_pb2.Mesh(
                data=Server_pb2.MeshData(
                    faces=np.ascontiguousarray(faces[slices]).tobytes()
                )
            )

    def _run_volume(self, volume_id, array, chunks, metadata):
        try:
            response = self._stub.IngestVolume(
                self._volume_iterator(volume_id, array, chunks, metadata)
            )
            if response.success:
                logger.info("Volume '%s' sent successfully", volume_id)
            else:
                logger.error("Volume '%s' failed: %s", volume_id, response.mesg)
        except Exception as e:
            logger.error("Volume '%s' error: %s", volume_id, e)


    def _run_mesh(self, mesh_id, vertices, faces, v_chunks, f_chunks):
        try:
            response = self._stub.IngestMesh(
                self._mesh_iterator(mesh_id, vertices, faces, v_chunks, f_chunks)
            )
            if response.success:
                logger.info("Mesh '%s' sent successfully", mesh_id)
            else:
                logger.error("Mesh '%s' failed: %s", mesh_id, response.mesg)
        except Exception as e:
            logger.error("Mesh '%s' error: %s", mesh_id, e)

    def _fire(self, target, args) -> threading.Thread:
        t = threading.Thread(target=target, args=args, daemon=True)
        self._pending.append(t)
        t.start()
        # clean up finished threads from the list automatically
        self._pending = [p for p in self._pending if p.is_alive()]
        return t

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_volume(
        self,
        volume_id: str,
        array:     np.ndarray,
        chunks:    tuple[int, ...] | None = None,
        metadata:  dict | None = None,
    ) -> None:
        if chunks is None:
            chunks = tuple(min(64, s) for s in array.shape)
        # no divisibility check — zarr handles partial edge blocks natively
        self._fire(self._run_volume, (volume_id, array, chunks, metadata))
        logger.debug("Volume '%s' queued for sending", volume_id)

    def send_mesh(
        self,
        mesh_id:  str,
        vertices: np.ndarray,
        faces:    np.ndarray,
        v_chunks: tuple[int, ...] | None = None,
        f_chunks: tuple[int, ...] | None = None,
    ) -> None:
        if v_chunks is None:
            v_chunks = tuple(min(512, s) for s in vertices.shape)
        if f_chunks is None:
            f_chunks = tuple(min(512, s) for s in faces.shape)

        for dim, (s, c) in enumerate(zip(vertices.shape, v_chunks)):
            if s % c != 0:
                raise ValueError(
                    f"vertices shape[{dim}]={s} not divisible by v_chunks[{dim}]={c}"
                )
        for dim, (s, c) in enumerate(zip(faces.shape, f_chunks)):
            if s % c != 0:
                raise ValueError(
                    f"faces shape[{dim}]={s} not divisible by f_chunks[{dim}]={c}"
                )

        self._fire(self._run_mesh, (mesh_id, vertices, faces, v_chunks, f_chunks))
        logger.debug("Mesh '%s' queued for sending", mesh_id)

    def wait(self, timeout: float | None = None) -> None:
        for t in list(self._pending):
            t.join(timeout=timeout)
        self._pending = [t for t in self._pending if t.is_alive()]

    def close(self) -> None:
        self.wait()
        self._channel.close()
        logger.info("DataClient closed")


def pad_to_chunks(arr: np.ndarray, chunks: tuple[int, ...]) -> np.ndarray:
    """Pad array so every dimension is divisible by the corresponding chunk size."""
    pad_width = [
        (0, (-s % c) % c)   # how many elements to add at the end of each dim
        for s, c in zip(arr.shape, chunks)
    ]
    if any(p[1] > 0 for p in pad_width):
        arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
    return arr


def prepare_nifti(path: str) -> np.ndarray:
    """
    Load a NIfTI and return a float32 array ready for napari.
    All orientation, resampling and axis ordering is applied here.
    The output is a plain (z, y, x) numpy array — no affine needed.
    """
    img = MedVol(path)

    # MedVol gives (x, y, z) oriented — napari expects (z, y, x)
    #arr = np.transpose(img.array, axes=(0, 1, 2)).astype(np.float32)
    arr = img.array
    return arr


def send_nifti(client, path, volume_id=None, chunks=None):
    import os

    arr = prepare_nifti(path)

    if volume_id is None:
        volume_id = os.path.splitext(os.path.basename(path))[0].replace(".nii", "")

    if chunks is None:
        chunks = tuple(min(64, s) for s in arr.shape)

    client.send_volume(
        volume_id = volume_id,
        array     = arr,
        chunks    = chunks,
    )
    print(f"'{path}' → '{volume_id}'  shape={arr.shape}  padded={arr.shape}")