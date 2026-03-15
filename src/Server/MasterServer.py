import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import grpc
import asyncio
from concurrent import futures
import time
from datetime import datetime

from generated import Server_pb2
from generated import Server_pb2_grpc

from grpc_reflection.v1alpha import reflection

from Server import ZarrService

class DataIngestServicer(Server_pb2_grpc.DataIngestServicer):
    """Basic implementation of the DataIngest service"""
    
    def __init__(self):
        self.volumes = {}
        self.meshes = {}
        self.subscribers = []
    
    async def _notify_subscribers(self, notification):
        """Send notification to all subscribers"""
        for queue in self.subscribers:
            await queue.put(notification)
    
    async def IngestVolume(self, request_iterator, context):
        current_volume = None
        
        async for chunk in request_iterator:
            if chunk.HasField('header'):
                current_volume = chunk.header.volume_id
                self.volumes[current_volume] = {
                    'metadata': chunk.header,
                    'data': b''
                }
            elif chunk.HasField('data'):
                self.volumes[current_volume]['data'] += chunk.data.data
        
        vol_data = self.volumes[current_volume]
        metadata = vol_data['metadata']
        
        volume_metadata = Server_pb2.VolumeMetadata(
            volume_id=metadata.volume_id,
            shape=metadata.shape,
            dtype=metadata.dtype,
            size_bytes=len(vol_data['data']),
            created_at=datetime.now().isoformat(),
            metadata=metadata.metadata
        )
        
        notification = Server_pb2.Notification(
            type=Server_pb2.ADDED,
            data_type=Server_pb2.VOLUME,
            id=current_volume,
            volume_metadata=volume_metadata
        )
        
        await self._notify_subscribers(notification)
        
        return Server_pb2.IngestVolumeResponse(
            success=True,
            mesg="Volume ingested successfully",
            volume_id=current_volume
        )
    
    async def IngestMesh(self, request_iterator, context):
        current_mesh = None
        
        async for chunk in request_iterator:
            if chunk.HasField('header'):
                current_mesh = chunk.header.mesh_id
                self.meshes[current_mesh] = {
                    'metadata': chunk.header,
                    'vertices': b'',
                    'faces': b''
                }
            elif chunk.HasField('data'):
                self.meshes[current_mesh]['vertices'] += chunk.data.vertices
                self.meshes[current_mesh]['faces'] += chunk.data.faces
        
        # Create notification for new mesh
        mesh_data = self.meshes[current_mesh]
        metadata = mesh_data['metadata']
        
        mesh_metadata = Server_pb2.MeshMetadata(
            Mesh_id=metadata.mesh_id,
            vertices_shape=metadata.vertices_shape,
            faces_shape=metadata.faces_shape,
            vertices_dtype=metadata.vertices_dtype,
            faces_dtype=metadata.faces_dtype,
            vertices_size_bytes=len(mesh_data['vertices']),
            faces_size_bytes=len(mesh_data['faces']),
            created_at=datetime.now().isoformat(),
            metadata=metadata.metadata
        )

        notification = Server_pb2.Notification(
            type=Server_pb2.ADDED,
            data_type=Server_pb2.MESH,
            id=current_mesh,
            mesh_metadta=mesh_metadata
        )
        
        # Notify subscribers
        await self._notify_subscribers(notification)
        return Server_pb2.IngestMeshResponse(
            success=True,
            mesg="Mesh ingested successfully",
            mesh_id=current_mesh
        )
    
    # List all data
    def ListData(self, request, context):
        response = Server_pb2.DataList()
        
        # Add volumes
        for vol_id, vol_data in self.volumes.items():
            metadata = vol_data['metadata']
            vol_metadata = response.volumes.add()
            vol_metadata.volume_id = metadata.volume_id
            vol_metadata.shape.extend(metadata.shape)
            vol_metadata.dtype = metadata.dtype
            vol_metadata.size_bytes = len(vol_data['data'])
            vol_metadata.created_at = datetime.now().isoformat()
            vol_metadata.metadata.update(metadata.metadata)
        
        # Add meshes
        for mesh_id, mesh_data in self.meshes.items():
            metadata = mesh_data['metadata']
            mesh_metadata = response.meshes.add()
            mesh_metadata.Mesh_id = metadata.mesh_id
            mesh_metadata.vertices_shape.extend(metadata.vertices_shape)
            mesh_metadata.faces_shape.extend(metadata.faces_shape)
            mesh_metadata.vertices_dtype = metadata.vertices_dtype
            mesh_metadata.faces_dtype = metadata.faces_dtype
            mesh_metadata.vertices_size_bytes = len(mesh_data['vertices'])
            mesh_metadata.faces_size_bytes = len(mesh_data['faces'])
            mesh_metadata.created_at = datetime.now().isoformat()
            mesh_metadata.metadata.update(metadata.metadata)
        
        return response
    
    def GetVolumeMetatdata(self, request, context):
        vol_data = self.volumes.get(request.id)
        if not vol_data:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Volume {request.id} not found")
            return Server_pb2.VolumeMetadata()
        
        metadata = vol_data['metadata']
        return Server_pb2.VolumeMetadata(
            volume_id=metadata.volume_id,
            shape=metadata.shape,
            dtype=metadata.dtype,
            size_bytes=len(vol_data['data']),
            created_at=datetime.now().isoformat(),
            metadata=metadata.metadata
        )
    
    def GetMeshMetatdata(self, request, context):
        mesh_data = self.meshes.get(request.id)
        if not mesh_data:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Mesh {request.id} not found")
            return Server_pb2.MeshMetadata()
        
        metadata = mesh_data['metadata']
        return Server_pb2.MeshMetadata(
            Mesh_id=metadata.mesh_id,
            vertices_shape=metadata.vertices_shape,
            faces_shape=metadata.faces_shape,
            vertices_dtype=metadata.vertices_dtype,
            faces_dtype=metadata.faces_dtype,
            vertices_size_bytes=len(mesh_data['vertices']),
            faces_size_bytes=len(mesh_data['faces']),
            created_at=datetime.now().isoformat(),
            metadata=metadata.metadata
        )
    
    async def FetchVolume(self, request, context):
        vol_data = self.volumes.get(request.id)
        if not vol_data:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return
        
        yield Server_pb2.Volume(header=vol_data['metadata'])
        
        chunk_size = 1024 * 64  # 64KB chunks
        data = vol_data['data']
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            yield Server_pb2.Volume(data=Server_pb2.VolumeData(data=chunk))
    
    async def FetchMesh(self, request, context):
        mesh_data = self.meshes.get(request.id)
        if not mesh_data:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            return
        
        yield Server_pb2.Mesh(header=mesh_data['metadata'])
        
        yield Server_pb2.Mesh(data=Server_pb2.MeshData(
            vertices=mesh_data['vertices'],
            faces=mesh_data['faces']
        ))
    
    async def SubscribeNotifications(self, request, context):
        queue = asyncio.Queue()
        self.subscribers.append(queue)
        
        try:
            while True:
                notification = await queue.get()
                yield notification
        finally:
            self.subscribers.remove(queue)

async def serve():
    """Start the gRPC server"""
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ]
    )
    
    Server_pb2_grpc.add_DataIngestServicer_to_server(
        DataIngestServicer(), server
    )

    SERVICE_NAMES = (
        Server_pb2.DESCRIPTOR.services_by_name['DataIngest'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    server.add_insecure_port('[::]:50051')
    
    print("Starting server on port 50051...")
    await server.start()
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        print("Shutting down...")
        await server.stop(0)

if __name__ == '__main__':
    asyncio.run(serve())
