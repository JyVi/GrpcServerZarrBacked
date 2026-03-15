import io
import zarr
import logging
import asyncio
import numpy as np
from typing import cast
from pprint import pprint
from datetime import datetime
from typing import Union, Optional, AsyncIterator

from zarr import create_hierarchy
from zarr.core.buffer import default_buffer_prototype
from zarr.core.group import GroupMetadata, ArrayV2Metadata, ArrayV3Metadata
from zarr.storage import MemoryStore, LocalStore
from zarr.experimental.cache_store import CacheStore

logger = logging.getLogger(__name__)

ZARR_ROOT         = "data/zarrlocalstorage.zarr"
VOLUME_GROUP      = "Volume"
MESH_GROUP        = "Mesh"
CACHE_SIZE_BYTES  = 512 * 1024 * 1024 # 512 MB

class ZarrService:
    def __init__(self, root_path=ZARR_ROOT, cache_size=CACHE_SIZE_BYTES):
        self.root_path = root_path
        self.cache_size = cache_size

        self._memory_store:    Optional[MemoryStore]  = None
        self._local_store:     Optional[LocalStore]   = None
        self._cached_store:    Optional[CacheStore]   = None
 
        self._root:         Optional[zarr.Group] = None
        self._volume_group: Optional[zarr.Group] = None
        self._mesh_group:   Optional[zarr.Group] = None
 
        self._volume_meta: dict[str, dict] = {}
        self._mesh_meta:   dict[str, dict] = {}
        self._volume_pending: dict[str, int] = {} # volume_id -> open array
        self._mesh_pending:   dict[str, dict[str, int]] = {} # mesh_id -> {"vertices": arr, "faces": arr}
    
    async def initialise(self) -> None:
        """Initialise the Zarr storage hierarchy and cache"""
        self._memory_store = MemoryStore()
        self._local_store = LocalStore(self.root_path)
        self._cached_store = CacheStore(
            store=self._memory_store,
            cache_store=self._local_store,
            max_size=self.cache_size
        )

        hierarchy_exists = await self._hierarchy_exists()
 
        if hierarchy_exists:
            logger.info("Existing zarr hierarchy detected - loading metadata …")
            await self._load_existing_hierarchy()
        else:
            logger.info("No existing hierarchy found - creating fresh layout …")
            await self._create_hierarchy()
 
        logger.info(
            "ZarrService ready  |  volumes=%d  meshes=%d",
            len(self._volume_meta), len(self._mesh_meta),
        )

    async def _hierarchy_exists(self) -> bool:
        """Return True when the Volume and Mesh groups already exist on disk."""
        assert self._local_store is not None, "_local_store not initialised"
        assert self._cached_store is not None, "_cached_store not initialised"
        vol_key  = f"{VOLUME_GROUP}/zarr.json"
        mesh_key = f"{MESH_GROUP}/zarr.json"
        try:
            vol_exists  = await self._local_store.get(vol_key,  prototype=default_buffer_prototype()) is not None
            mesh_exists = await self._local_store.get(mesh_key, prototype=default_buffer_prototype()) is not None
            return vol_exists and mesh_exists
        except Exception:
            return False
 
    async def _create_hierarchy(self) -> None:
        """Build the Volume / Mesh group layout from scratch."""
        node_spec: dict[str, GroupMetadata | ArrayV2Metadata | ArrayV3Metadata]  = {
            VOLUME_GROUP: GroupMetadata(),
            MESH_GROUP:   GroupMetadata(),
        }
        assert self._cached_store is not None, "_cached_store not initialised, call initialise() first"
        create_hierarchy(store=self._cached_store, nodes=node_spec)
 
        self._root         = zarr.open_group(store=self._cached_store, mode="a")
        self._volume_group = self._root.require_group(VOLUME_GROUP)
        self._mesh_group   = self._root.require_group(MESH_GROUP)
 
    async def _load_existing_hierarchy(self) -> None:
        assert self._local_store   is not None, "_local_store not initialised"
        assert self._cached_store  is not None, "_cached_store not initialised"

        # open on local store to read what's actually on disk
        root_local         = zarr.open_group(store=self._local_store, mode="r")
        volume_group_local = root_local.require_group(VOLUME_GROUP)
        mesh_group_local   = root_local.require_group(MESH_GROUP)

        for vol_id in volume_group_local.group_keys():
            grp   = volume_group_local[vol_id]
            assert isinstance(grp, zarr.Group)
            attrs = dict(grp.attrs)

            if attrs:
                self._volume_meta[vol_id] = attrs
            else:
                node = grp.get("data")
                if node is not None and isinstance(node, zarr.Array):
                    logger.warning("Volume '%s' has no attrs — reconstructing from array", vol_id)
                    self._volume_meta[vol_id] = {
                        "volume_id":  vol_id,
                        "shape":      list(node.shape),
                        "dtype":      str(node.dtype),
                        "chunkshape": list(node.chunks),
                        "size_bytes": node.nbytes,
                        "created_at": "",
                        "metadata":   {},
                    }
                else:
                    logger.warning("Volume '%s' is empty — skipping", vol_id)

        for mesh_id in mesh_group_local.group_keys():
            grp   = mesh_group_local[mesh_id]
            assert isinstance(grp, zarr.Group)
            attrs = dict(grp.attrs)

            if attrs:
                self._mesh_meta[mesh_id] = attrs
            else:
                v_node = grp.get("vertices")
                f_node = grp.get("faces")
                if (v_node is not None and isinstance(v_node, zarr.Array) and
                    f_node is not None and isinstance(f_node, zarr.Array)):
                    logger.warning("Mesh '%s' has no attrs — reconstructing from arrays", mesh_id)
                    self._mesh_meta[mesh_id] = {
                        "mesh_id":             mesh_id,
                        "vertices_shape":      list(v_node.shape),
                        "faces_shape":         list(f_node.shape),
                        "vertices_dtype":      str(v_node.dtype),
                        "faces_dtype":         str(f_node.dtype),
                        "vertices_chunkshape": list(v_node.chunks),
                        "faces_chunkshape":    list(f_node.chunks),
                        "vertices_size_bytes": v_node.nbytes,
                        "faces_size_bytes":    f_node.nbytes,
                        "created_at":          "",
                        "metadata":            {},
                    }
                else:
                    logger.warning("Mesh '%s' is empty — skipping", mesh_id)

        # now open on cached_store for all runtime operations
        # the cache will warm naturally as chunks are read/written
        self._root         = zarr.open_group(store=self._cached_store, mode="a")
        self._volume_group = self._root.require_group(VOLUME_GROUP)
        self._mesh_group   = self._root.require_group(MESH_GROUP)

    async def write_volume(
        self,
        volume_id:  str,
        data:       bytes,
        shape:      list[int],
        dtype:      str,
        datetime:    str = datetime.now().isoformat(),
        metadata:   dict[str, str] | None = None,
    ) -> dict:
        assert self._volume_group is not None, "_volume_group not initialised, call initialise() first"
        
        arr = np.frombuffer(data, dtype=dtype).reshape(shape)
        vol_group = self._volume_group.require_group(volume_id)
 
        # coalese chunks
        chunks = tuple(min(s, 64) for s in shape)
 
        if "data" in vol_group:
            del vol_group["data"]
 
        z_arr = vol_group.create_array(
            name="data",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
        )
        z_arr[:] = arr
 
        meta = {
            "volume_id":   volume_id,
            "shape":       shape,
            "dtype":       dtype,
            "size_bytes":  len(data),
            "created_at":  datetime,
            "chunkshape": list(z_arr.chunks),
            "metadata":    metadata or {},
        }
 
        # Persist metadata as zarr group attributes
        vol_group.attrs.update(meta)
 
        self._volume_meta[volume_id] = meta
        logger.debug("Volume %s written  shape=%s  dtype=%s", volume_id, shape, dtype)
        return meta


    async def prepare_volume(
        self,
        volume_id: str,
        shape:     list[int],
        dtype:     str,
        chunks:    list[int],       # now comes from client via header
        metadata:  dict[str, str] | None = None,
    ) -> None:
        assert self._volume_group is not None, "_volume_group not initialised"
        assert self._volume_pending is not None, "_volume_pending not initialised"
        vol_group = self._volume_group.require_group(volume_id)

        vol_group.create_array(
            name="data",
            shape=shape,
            chunks=tuple(chunks),
            dtype=dtype,
            overwrite=True,
        )

        # block_cursor starts at 0 — counts zarr blocks not bytes
        self._volume_pending[volume_id] = 0

        logger.debug("Volume '%s' prepared  shape=%s  chunks=%s  dtype=%s", volume_id, shape, chunks, dtype)

    async def stream_volume_chunk(self, volume_id: str, chunk_data: bytes) -> None:
        if volume_id not in self._volume_pending:
            logger.warning("stream_volume_chunk: no pending volume '%s'", volume_id)
            return
        assert self._volume_group is not None, "_volume_group not initialised"

        vol_node = self._volume_group[volume_id]
        assert isinstance(vol_node, zarr.Group)
        z_arr = vol_node["data"]
        assert isinstance(z_arr, zarr.Array)

        block_cursor = self._volume_pending[volume_id]

        # total number of blocks per dimension
        n_blocks = [s // c for s, c in zip(z_arr.shape, z_arr.chunks)]

        # convert flat block counter to nd block index
        block_idx = cast(tuple[int, ...], np.unravel_index(block_cursor, n_blocks))
        # actual shape of this block (edge blocks may be smaller)
        block_shape = tuple(
            min(c, s - b * c)
            for b, c, s in zip(block_idx, z_arr.chunks, z_arr.shape)
        )

        chunk_arr = np.frombuffer(chunk_data, dtype=z_arr.dtype).reshape(block_shape)

        # pure write — no read needed because chunk is perfectly aligned
        z_arr.set_block_selection(block_idx, chunk_arr)

        self._volume_pending[volume_id] = block_cursor + 1
        logger.debug("Volume '%s' block %s written", volume_id, block_idx)

    async def finalise_volume(
        self,
        volume_id: str,
        created_at: str = datetime.now().isoformat(),
        metadata:   dict[str, str] | None = None,
    ) -> dict | None:
        if volume_id not in self._volume_pending:
            logger.warning("finalise_volume: no pending volume '%s'", volume_id)
            return None

        assert self._volume_group is not None, "_volume_group not initialised"

        self._volume_pending.pop(volume_id)

        vol_node = self._volume_group[volume_id]
        assert isinstance(vol_node, zarr.Group)
        z_arr = vol_node["data"]
        assert isinstance(z_arr, zarr.Array)

        meta = {
            "volume_id":  volume_id,
            "shape":      list(z_arr.shape),
            "dtype":      str(z_arr.dtype),
            "chunks":     list(z_arr.chunks),
            "size_bytes": z_arr.nbytes,
            "created_at": created_at,
            "metadata":   metadata or {},
        }
        vol_node.attrs.update(meta)
        self._volume_meta[volume_id] = meta

        logger.debug("Volume '%s' finalised", volume_id)
        return meta


    async def read_volume(self, volume_id: str) -> tuple[np.ndarray, dict] | None:
        if volume_id not in self._volume_meta:
            logger.warning("read_volume: volume '%s' not found", volume_id)
            return None
        assert self._volume_group is not None, "_volume_group not initialised, call initialise() first"
        
        volume = self._volume_group[volume_id]
        assert isinstance(volume, zarr.Group), f"Expected Group for volume '{volume_id}', got {type(volume)}"
        z_arr = volume["data"]
        assert isinstance(z_arr, zarr.Array), f"Expected Array for volume '{volume_id}', got {type(z_arr)}"
        return cast(np.ndarray, z_arr[:]), self._volume_meta[volume_id]

    async def read_volume_chunks(self, volume_id: str) -> AsyncIterator[np.ndarray]:
        if volume_id not in self._volume_meta:
            logger.warning("read_volume_chunks: volume '%s' not found", volume_id)
            return
        assert self._volume_group is not None, "_volume_group not initialised, call initialise() first"
        assert isinstance(self._volume_group[volume_id], zarr.Group), f"Expected Group for volume '{volume_id}', got {type(self._volume_group[volume_id])}"
        
        volume = self._volume_group[volume_id]
        assert isinstance(volume, zarr.Group), f"Expected Group for volume '{volume_id}', got {type(volume)}"
        z_arr = volume["data"]
        assert isinstance(z_arr, zarr.Array), f"Expected Array for volume '{volume_id}', got {type(z_arr)}"
        
        block_counts = [s // c for s, c in zip(z_arr.shape, z_arr.chunks)]
        for block_idx in np.ndindex(*block_counts):
            yield z_arr.get_block_selection(block_idx)
 
    def get_volume_metadata(self, volume_id: str) -> dict | None:
        if volume_id not in self._volume_meta:
            logger.warning("get_volume_metadata: volume '%s' not found", volume_id)
            return None
        return self._volume_meta[volume_id]
    
    def list_volume_ids(self) -> list[str]:
        return list(self._volume_meta.keys())

    def list_volumes(self) -> list[dict]:
        return list(self._volume_meta.values())


    async def write_mesh(
        self,
        mesh_id:        str,
        vertices:       bytes,
        faces:          bytes,
        vertices_shape: list[int],
        faces_shape:    list[int],
        vertices_dtype: str,
        faces_dtype:    str,
        created_at:     str = datetime.now().isoformat(),
        metadata:       dict[str, str] | None = None,
    ) -> dict:
        assert self._mesh_group is not None, "_mesh_group not initialised, call initialise() first"
        vert_arr   = np.frombuffer(vertices, dtype=vertices_dtype).reshape(vertices_shape)
        face_arr   = np.frombuffer(faces,    dtype=faces_dtype   ).reshape(faces_shape)
        mesh_group = self._mesh_group.require_group(mesh_id)

        vert_chunks = tuple(min(s, 512) for s in vertices_shape)
        face_chunks = tuple(min(s, 512) for s in faces_shape)

        if "vertices" in mesh_group:
            del mesh_group["vertices"]

        if "faces" in mesh_group:
            del mesh_group["faces"]

        z_vert = mesh_group.create_array(
            name="vertices",
            shape=vertices_shape,
            chunks=vert_chunks,
            dtype=vertices_dtype,
        )
        z_vert[:] = vert_arr

        z_face = mesh_group.create_array(
            name="faces",
            shape=faces_shape,
            chunks=face_chunks,
            dtype=faces_dtype,
        )
        z_face[:] = face_arr

        meta = {
            "mesh_id":             mesh_id,
            "vertices_shape":      vertices_shape,
            "faces_shape":         faces_shape,
            "vertices_dtype":      vertices_dtype,
            "faces_dtype":         faces_dtype,
            "vertices_size_bytes": len(vertices),
            "faces_size_bytes":    len(faces),
            "created_at":          created_at,
            "vertices_chunkshape": list(z_vert.chunks),
            "faces_chunkshape":    list(z_face.chunks),
            "metadata":            metadata or {},
        }
        mesh_group.attrs.update(meta)
        self._mesh_meta[mesh_id] = meta
        logger.debug("Mesh %s written  vertices=%s  faces=%s", mesh_id, vertices_shape, faces_shape)
        return meta

    async def prepare_mesh(
        self,
        mesh_id:         str,
        vertices_shape:  list[int],
        faces_shape:     list[int],
        vertices_dtype:  str,
        faces_dtype:     str,
        vertices_chunks: list[int],   # from header
        faces_chunks:    list[int],   # from header
        metadata:        dict[str, str] | None = None,
    ) -> None:
        assert self._mesh_group is not None, "_mesh_group not initialised"

        mesh_group = self._mesh_group.require_group(mesh_id)

        mesh_group.create_array(
            name="vertices",
            shape=vertices_shape,
            chunks=tuple(vertices_chunks),
            dtype=vertices_dtype,
            overwrite=True,
        )
        mesh_group.create_array(
            name="faces",
            shape=faces_shape,
            chunks=tuple(faces_chunks),
            dtype=faces_dtype,
            overwrite=True,
        )

        # one block counter per array
        self._mesh_pending[mesh_id] = {"vertices": 0, "faces": 0}

        logger.debug("Mesh '%s' prepared", mesh_id)

    async def stream_mesh_chunk(
        self,
        mesh_id:    str,
        array_name: str,
        chunk_data: bytes,
    ) -> None:
        if mesh_id not in self._mesh_pending:
            logger.warning("stream_mesh_chunk: no pending mesh '%s'", mesh_id)
            return

        assert self._mesh_group is not None, "_mesh_group not initialised"

        mesh_node = self._mesh_group[mesh_id]
        assert isinstance(mesh_node, zarr.Group)
        z_arr = mesh_node[array_name]
        assert isinstance(z_arr, zarr.Array)

        block_cursor = self._mesh_pending[mesh_id][array_name]
        n_blocks     = [s // c for s, c in zip(z_arr.shape, z_arr.chunks)]
        block_idx = cast(tuple[int, ...], np.unravel_index(block_cursor, n_blocks))
        block_shape  = tuple(
            min(c, s - b * c)
            for b, c, s in zip(block_idx, z_arr.chunks, z_arr.shape)
        )

        chunk_arr = np.frombuffer(chunk_data, dtype=z_arr.dtype).reshape(block_shape)
        z_arr.set_block_selection(block_idx, chunk_arr)

        self._mesh_pending[mesh_id][array_name] = block_cursor + 1
        logger.debug("Mesh '%s' %s block %s written", mesh_id, array_name, block_idx)

    async def finalise_mesh(
        self,
        mesh_id:    str,
        created_at: str = datetime.now().isoformat(),
        metadata:   dict[str, str] | None = None,
    ) -> dict | None:
        if mesh_id not in self._mesh_pending:
            logger.warning("finalise_mesh: no pending mesh '%s'", mesh_id)
            return None

        assert self._mesh_group is not None, "_mesh_group not initialised"

        self._mesh_pending.pop(mesh_id)

        mesh_node = self._mesh_group[mesh_id]
        assert isinstance(mesh_node, zarr.Group), f"Expected Group for '{mesh_id}'"

        z_vert = mesh_node["vertices"]
        z_face = mesh_node["faces"]
        assert isinstance(z_vert, zarr.Array), f"Expected Array for vertices in '{mesh_id}'"
        assert isinstance(z_face, zarr.Array), f"Expected Array for faces in '{mesh_id}'"

        meta = {
            "mesh_id":              mesh_id,
            "vertices_shape":       list(z_vert.shape),
            "faces_shape":          list(z_face.shape),
            "vertices_dtype":       str(z_vert.dtype),
            "faces_dtype":          str(z_face.dtype),
            "vertices_size_bytes":  z_vert.nbytes,
            "faces_size_bytes":     z_face.nbytes,
            "vertices_chunkshape":  list(z_vert.chunks),
            "faces_chunkshape":     list(z_face.chunks),
            "created_at":           created_at,
            "metadata":             metadata or {},
        }
        mesh_node.attrs.update(meta)
        self._mesh_meta[mesh_id] = meta

        logger.debug("Mesh '%s' finalised", mesh_id)
        return meta


    async def read_mesh(self, mesh_id: str) -> tuple[np.ndarray, np.ndarray, dict] | None:
        if mesh_id not in self._mesh_meta:
            logger.warning("read_mesh: mesh '%s' not found", mesh_id)
            return None

        assert self._mesh_group is not None, "_mesh_group not initialised, call initialise() first"
        mesh_group = self._mesh_group[mesh_id]
        assert isinstance(mesh_group, zarr.Group), f"Expected Group for mesh '{mesh_id}', got {type(mesh_group)}"
        vertices = mesh_group["vertices"]
        faces    = mesh_group["faces"]
        assert isinstance(vertices, zarr.Array), f"Expected Array for mesh '{mesh_id}' vertices, got {type(vertices)}"
        assert isinstance(faces, zarr.Array), f"Expected Array for mesh '{mesh_id}' faces, got {type(faces)}"
        return cast(np.ndarray, vertices[:]), cast(np.ndarray, faces[:]), self._mesh_meta[mesh_id]

    async def read_mesh_chunks(self, mesh_id: str) -> AsyncIterator[tuple[str, np.ndarray]]:
        if mesh_id not in self._mesh_meta:
            logger.warning("read_mesh_chunks: mesh '%s' not found", mesh_id)
            return
        assert self._mesh_group is not None, "_mesh_group not initialised, call initialise() first"

        mesh_group = self._mesh_group[mesh_id]
        assert isinstance(mesh_group, zarr.Group), f"Expected Group for mesh '{mesh_id}', got {type(mesh_group)}"
        for name in ("vertices", "faces"):
            z_arr        = mesh_group[name]
            assert isinstance(z_arr, zarr.Array), f"Expected Array for mesh '{mesh_id}' {name}, got {type(z_arr)}"
            block_counts = [s // c for s, c in zip(z_arr.shape, z_arr.chunks)]

            for block_idx in np.ndindex(*block_counts):
                yield name, z_arr.get_block_selection(block_idx)

    def get_mesh_metadata(self, mesh_id: str) -> dict | None:
        if mesh_id not in self._mesh_meta:
            logger.warning("get_mesh_metadata: mesh '%s' not found", mesh_id)
            return None
        return self._mesh_meta[mesh_id]

    def list_mesh_ids(self) -> list[str]:
        return list(self._mesh_meta.keys())

    def list_meshes(self) -> list[dict]:
        return list(self._mesh_meta.values())



def test_zarr_service():
    memory_store = MemoryStore()
    local_store_path = "data/zarr_cache.zarr"
    persistent_cache = LocalStore(local_store_path)
    cached_store = CacheStore(
        store=memory_store, # In-memory cache
        cache_store=persistent_cache,
        max_size=256*1024*1024 # 256MB cache
    )

    Root = zarr.group(store=cached_store)
    Volume = zarr.open_group('data/zarr_cache.zarr/volume', mode='w')
    Mesh = zarr.open_group('data/zarr_cache.zarr/mesh', mode='w')

    print(Root.tree())
    print(Root.info)
    print(Root.info_complete())
    print("Cache store info:")
    pprint(cached_store.cache_info())
    pprint(cached_store.cache_stats())

    volArray = Volume.create_array(name='data', shape=(100, 100, 100), chunks=(70, 70, 70), dtype='float32')
    vert_array = Mesh.create_array(name='vertices', shape=(1000, 3), chunks=(700, 3), dtype='float32')
    face_array = Mesh.create_array(name='faces', shape=(500, 3), chunks=(400, 3), dtype='int32')

    volArray[:] = 1.0
    vert_array[:] = 0.5
    face_array[:] = 1
    pprint(Root.tree())

    node_spec : dict[str, Union[GroupMetadata, ArrayV2Metadata, ArrayV3Metadata]] = {'ServerData.zarr/Volume': GroupMetadata(), 'ServerData.zarr/Mesh': GroupMetadata()}

    async def checkingHistory():
        print()
        print(await cached_store.get('ServerData.zarr', prototype=default_buffer_prototype()))  # Check if 'Volume' node exists in cache store
        print()
        if await cached_store.get('Volume', prototype=default_buffer_prototype()):
            print("\nVolume node exists in cache store\n")
            nodes_created = zarr.open_group(store=cached_store, mode='r')
        else:
            print("\nnodes doesnt exists\n")
            nodes_created = dict(create_hierarchy(store=cached_store, nodes=node_spec))
        return nodes_created

    nodes_created = asyncio.run(checkingHistory())

    print("\nHierarchy created:\n")
    pprint(nodes_created)
    print("Cache store info after hierarchy creation:")
    pprint(cached_store.cache_info())
    pprint(cached_store.cache_stats())
    pprint("Root tree after hierarchy creation:")

    output = io.StringIO()
    pprint(nodes_created, stream=output, width=60)
    print(output.getvalue())

    print("\nAccessing volume node:\n")
    volumenode = nodes_created["ServerData.zarr/Volume"]
    print(volumenode.info)
    print(volumenode.info_complete())
    print(volumenode.tree())
    vnode = volumenode.create_array(name='Vol_001', shape=(100, 100, 100), chunks=(70, 70, 70), dtype='float32')
    print(vnode)
    vnode[:] = 1.0
    vnode[:] = 4.0

    print(vnode[:])