from __future__ import annotations
import inspect
import functools
from typing import Iterable, ClassVar
import uuid
import numpy as np


class UniquelyIdentifiable:
    """
    Instances of this class can be identified via an UUID.
    This UUID is based on the class and the instance dictionary. It will stay
    the same after re-instantiation with the same parameters.

    Please note: Behavioral changes due to changing non-uuid base class
                 implementations are not catched by a new UUID.

    :cvar neutral_uuid: The eins of the `concat_uuid` operation.
    """

    neutral_uuid: ClassVar[uuid.UUID] = \
        uuid.UUID(bytes=bytes(np.eye(4, dtype=np.uint8)))

    @classmethod
    def _get_class_uuid(cls) -> uuid.UUID:
        """
        Unique identifier of the class that is based on its source code.
        """
        source_uuid = uuid.uuid5(uuid.UUID(int=0), inspect.getsource(cls))
        base_uuids = []
        for base in cls.__bases__:
            if issubclass(base, UniquelyIdentifiable):
                base_uuids.append(str(base._get_class_uuid()))  # pylint: disable=no-member,protected-access

        return functools.reduce(uuid.uuid5, base_uuids, source_uuid)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} uuid={self.uuid}>"

    @property
    def uuid(self) -> uuid.UUID:
        """
        Unique identifier of the instance based on the uuid of its class
        and the instance dictionary.
        """
        return uuid.uuid5(self._get_class_uuid(), repr(self.__dict__))

    @classmethod
    def concat_uuids(cls, uuids: Iterable[uuid.UUID]) -> uuid.UUID:
        """
        Concatenate the provided UUIDs to a new one.
        This operation is associative, but does not commute.

        :param uuids: UUIDs to concatenate

        :returns: The concatenated uuid. Defaults to :cvar:`neutral_uuid` for
            empty :param:`uuids`.
        """
        def _to_numpy(uid: uuid.UUID) -> np.ndarray:
            arr = np.frombuffer(uid.bytes, dtype=np.uint8)
            return arr.astype(np.uint32).reshape(4, 4)

        def _concat_uuids(id1, id2):
            out = np.matmul(_to_numpy(id1), _to_numpy(id2))
            out[out >= 256] %= 251  # overflow modulo largest prime <= 256
            return uuid.UUID(bytes=bytes(out.astype(np.uint8)))

        return functools.reduce(_concat_uuids, uuids, cls.neutral_uuid)
