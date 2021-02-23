"""
Tests of the data module.
"""
import unittest
import uuid

from timedatasets.common import UniquelyIdentifiable


class TestCommon(unittest.TestCase):
    """
    Test the common module.
    """

    def test_uniquely_identifiable(self):
        """
        Test the UniquelyIdentifiable class.
        """

        class Test1(UniquelyIdentifiable):
            def __init__(self, arg=1):
                self.arg = arg

        class Test2(UniquelyIdentifiable):
            def __init__(self, arg):
                self.arg = arg

        t11 = Test1()

        self.assertEqual(t11.uuid, Test1(1).uuid)
        self.assertNotEqual(t11.uuid, Test1(2).uuid)
        self.assertNotEqual(t11.uuid, Test2(1).uuid)
        self.assertIn(str(t11.uuid), repr(t11))

    def test_concat_uuids(self):
        """
        Test the concatenation of UUIDs.
        """
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()
        id3 = uuid.uuid4()
        self.assertNotEqual(UniquelyIdentifiable.concat_uuids([id1, id2]),
                            UniquelyIdentifiable.concat_uuids([id2, id1]))
        self.assertEqual(
            UniquelyIdentifiable.concat_uuids([id1, id2, id3]),
            UniquelyIdentifiable.concat_uuids([
                id1, UniquelyIdentifiable.concat_uuids([id2, id3])]))
        self.assertEqual(
            UniquelyIdentifiable.concat_uuids([id1, id2, id3]),
            UniquelyIdentifiable.concat_uuids([
                UniquelyIdentifiable.concat_uuids([id1, id2]), id3]))

        for _ in range(1000):
            id4 = uuid.uuid4()
            for _ in range(20):
                id_new = UniquelyIdentifiable.concat_uuids([id4, id4])
                self.assertNotEqual(id4, id_new,
                                    f"{id4} is the same as its square.")
                id4 = id_new


if __name__ == "__main__":
    unittest.main()
