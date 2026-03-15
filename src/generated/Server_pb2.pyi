from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VOLUME: _ClassVar[DataType]
    MESH: _ClassVar[DataType]

class NotificationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ADDED: _ClassVar[NotificationType]
    UPDATED: _ClassVar[NotificationType]
    DELETED: _ClassVar[NotificationType]
VOLUME: DataType
MESH: DataType
ADDED: NotificationType
UPDATED: NotificationType
DELETED: NotificationType

class VolumeHeader(_message.Message):
    __slots__ = ("volume_id", "shape", "dtype", "chunkshape", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VOLUME_ID_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    CHUNKSHAPE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    volume_id: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    chunkshape: _containers.RepeatedScalarFieldContainer[int]
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, volume_id: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., chunkshape: _Optional[_Iterable[int]] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class VolumeData(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class Volume(_message.Message):
    __slots__ = ("header", "data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    header: VolumeHeader
    data: VolumeData
    def __init__(self, header: _Optional[_Union[VolumeHeader, _Mapping]] = ..., data: _Optional[_Union[VolumeData, _Mapping]] = ...) -> None: ...

class MeshHeader(_message.Message):
    __slots__ = ("mesh_id", "vertices_shape", "faces_shape", "vertices_dtype", "faces_dtype", "vertices_chunkshape", "faces_chunkshape", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MESH_ID_FIELD_NUMBER: _ClassVar[int]
    VERTICES_SHAPE_FIELD_NUMBER: _ClassVar[int]
    FACES_SHAPE_FIELD_NUMBER: _ClassVar[int]
    VERTICES_DTYPE_FIELD_NUMBER: _ClassVar[int]
    FACES_DTYPE_FIELD_NUMBER: _ClassVar[int]
    VERTICES_CHUNKSHAPE_FIELD_NUMBER: _ClassVar[int]
    FACES_CHUNKSHAPE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    mesh_id: str
    vertices_shape: _containers.RepeatedScalarFieldContainer[int]
    faces_shape: _containers.RepeatedScalarFieldContainer[int]
    vertices_dtype: str
    faces_dtype: str
    vertices_chunkshape: _containers.RepeatedScalarFieldContainer[int]
    faces_chunkshape: _containers.RepeatedScalarFieldContainer[int]
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, mesh_id: _Optional[str] = ..., vertices_shape: _Optional[_Iterable[int]] = ..., faces_shape: _Optional[_Iterable[int]] = ..., vertices_dtype: _Optional[str] = ..., faces_dtype: _Optional[str] = ..., vertices_chunkshape: _Optional[_Iterable[int]] = ..., faces_chunkshape: _Optional[_Iterable[int]] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class MeshData(_message.Message):
    __slots__ = ("vertices", "faces")
    VERTICES_FIELD_NUMBER: _ClassVar[int]
    FACES_FIELD_NUMBER: _ClassVar[int]
    vertices: bytes
    faces: bytes
    def __init__(self, vertices: _Optional[bytes] = ..., faces: _Optional[bytes] = ...) -> None: ...

class Mesh(_message.Message):
    __slots__ = ("header", "data")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    header: MeshHeader
    data: MeshData
    def __init__(self, header: _Optional[_Union[MeshHeader, _Mapping]] = ..., data: _Optional[_Union[MeshData, _Mapping]] = ...) -> None: ...

class IngestVolumeResponse(_message.Message):
    __slots__ = ("success", "mesg", "volume_id")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESG_FIELD_NUMBER: _ClassVar[int]
    VOLUME_ID_FIELD_NUMBER: _ClassVar[int]
    success: bool
    mesg: str
    volume_id: str
    def __init__(self, success: bool = ..., mesg: _Optional[str] = ..., volume_id: _Optional[str] = ...) -> None: ...

class IngestMeshResponse(_message.Message):
    __slots__ = ("success", "mesg", "mesh_id")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESG_FIELD_NUMBER: _ClassVar[int]
    MESH_ID_FIELD_NUMBER: _ClassVar[int]
    success: bool
    mesg: str
    mesh_id: str
    def __init__(self, success: bool = ..., mesg: _Optional[str] = ..., mesh_id: _Optional[str] = ...) -> None: ...

class VolumeMetadata(_message.Message):
    __slots__ = ("volume_id", "shape", "dtype", "size_bytes", "created_at", "chunkshape", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    VOLUME_ID_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    CHUNKSHAPE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    volume_id: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    dtype: str
    size_bytes: int
    created_at: str
    chunkshape: _containers.RepeatedScalarFieldContainer[int]
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, volume_id: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., dtype: _Optional[str] = ..., size_bytes: _Optional[int] = ..., created_at: _Optional[str] = ..., chunkshape: _Optional[_Iterable[int]] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class MeshMetadata(_message.Message):
    __slots__ = ("mesh_id", "vertices_shape", "faces_shape", "vertices_dtype", "faces_dtype", "vertices_size_bytes", "faces_size_bytes", "created_at", "vertices_chunkshape", "faces_chunkshape", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MESH_ID_FIELD_NUMBER: _ClassVar[int]
    VERTICES_SHAPE_FIELD_NUMBER: _ClassVar[int]
    FACES_SHAPE_FIELD_NUMBER: _ClassVar[int]
    VERTICES_DTYPE_FIELD_NUMBER: _ClassVar[int]
    FACES_DTYPE_FIELD_NUMBER: _ClassVar[int]
    VERTICES_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    FACES_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    VERTICES_CHUNKSHAPE_FIELD_NUMBER: _ClassVar[int]
    FACES_CHUNKSHAPE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    mesh_id: str
    vertices_shape: _containers.RepeatedScalarFieldContainer[int]
    faces_shape: _containers.RepeatedScalarFieldContainer[int]
    vertices_dtype: str
    faces_dtype: str
    vertices_size_bytes: int
    faces_size_bytes: int
    created_at: str
    vertices_chunkshape: _containers.RepeatedScalarFieldContainer[int]
    faces_chunkshape: _containers.RepeatedScalarFieldContainer[int]
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, mesh_id: _Optional[str] = ..., vertices_shape: _Optional[_Iterable[int]] = ..., faces_shape: _Optional[_Iterable[int]] = ..., vertices_dtype: _Optional[str] = ..., faces_dtype: _Optional[str] = ..., vertices_size_bytes: _Optional[int] = ..., faces_size_bytes: _Optional[int] = ..., created_at: _Optional[str] = ..., vertices_chunkshape: _Optional[_Iterable[int]] = ..., faces_chunkshape: _Optional[_Iterable[int]] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class DataList(_message.Message):
    __slots__ = ("volumes", "meshes")
    VOLUMES_FIELD_NUMBER: _ClassVar[int]
    MESHES_FIELD_NUMBER: _ClassVar[int]
    volumes: _containers.RepeatedCompositeFieldContainer[VolumeMetadata]
    meshes: _containers.RepeatedCompositeFieldContainer[MeshMetadata]
    def __init__(self, volumes: _Optional[_Iterable[_Union[VolumeMetadata, _Mapping]]] = ..., meshes: _Optional[_Iterable[_Union[MeshMetadata, _Mapping]]] = ...) -> None: ...

class DataRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class Notification(_message.Message):
    __slots__ = ("type", "data_type", "id", "volume_metadata", "mesh_metadata")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    VOLUME_METADATA_FIELD_NUMBER: _ClassVar[int]
    MESH_METADATA_FIELD_NUMBER: _ClassVar[int]
    type: NotificationType
    data_type: DataType
    id: str
    volume_metadata: VolumeMetadata
    mesh_metadata: MeshMetadata
    def __init__(self, type: _Optional[_Union[NotificationType, str]] = ..., data_type: _Optional[_Union[DataType, str]] = ..., id: _Optional[str] = ..., volume_metadata: _Optional[_Union[VolumeMetadata, _Mapping]] = ..., mesh_metadata: _Optional[_Union[MeshMetadata, _Mapping]] = ...) -> None: ...

class NotificationRequest(_message.Message):
    __slots__ = ("auto_download", "data_types")
    AUTO_DOWNLOAD_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPES_FIELD_NUMBER: _ClassVar[int]
    auto_download: bool
    data_types: _containers.RepeatedScalarFieldContainer[DataType]
    def __init__(self, auto_download: bool = ..., data_types: _Optional[_Iterable[_Union[DataType, str]]] = ...) -> None: ...

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...
