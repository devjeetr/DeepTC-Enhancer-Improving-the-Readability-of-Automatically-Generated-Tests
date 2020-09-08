import os
import pickle


class FileReader:
    """A utility class used for random access to large files,
       that cannot be loaded into memory.
    """

    def __init__(self, filename, line_cache=None):
        if not os.path.exists(filename) and os.path.isfile(filename):
            raise Exception("Not a valid filename")

        self.filepath = filename
        file = open(filename, "r")
        self.line_offsets = {}

        if line_cache and os.path.exists(line_cache):
            with open(line_cache, "rb") as f:
                self.line_offsets = pickle.load(f)
            print("Found linecache")
        else:
            print("line cache not found")
            offset = 0
            line = file.readline()
            line_number = 0

            while line:
                self.line_offsets[line_number] = offset
                offset = offset + len(line)
                line = file.readline()
                line_number += 1

            if line_cache:
                with open(line_cache, "wb") as f:
                    pickle.dump(self.line_offsets, f)

        file.close()

    def __getitem__(self, idx):
        with open(self.filepath, "r") as f:
            if isinstance(idx, slice):
                idx = list(range(idx.start or 0, idx.stop or len(self), idx.step or 1))
                # return [self[i] for i in idx]
                data = []
                for i in idx:
                    f.seek(0)
                    f.seek(self.line_offsets[i])

                    data.append(f.readline().rstrip())
                return data
            else:
                f.seek(0)
                f.seek(self.line_offsets[idx])

                return f.readline().rstrip()

    def __len__(self):
        return len(self.line_offsets)

