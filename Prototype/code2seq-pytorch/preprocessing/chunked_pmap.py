import os
import sys
import time

import cursor
from humanfriendly import format_size, format_timespan
from multiprocessing import Pool

def chunk_list(seq, size):
    return (seq[i::size] for i in range(size))


def chunked_pmap(
        infile: str, mapper, on_receive_chunk=None, n_futures=12, file_chunk_size=2 ** 29, reducer=None
):
    """ Applies the given mapper to the the lines of the given infile,
        in chunks. Useful for when infile is too large to load into memory,
        and needs to be processed line by line, in parallel.

        :param infile the file to apply the chunked processing to
        :param mapper a function decorated with ray.remote that takes as
               its parameters a list of strings, representing a chunk of
               infile line by line
        :on_receive_chunk a function that ingests the output of mapper
        :param file_chunk_size size in bytes for each chunk
    """

    bytes_processed = 0
    file_size = os.path.getsize(infile)

    cursor.hide()
    start_time = time.time()

    mode = None
    if on_receive_chunk and not reducer:
        mode = "STREAM"
    elif reducer and not on_receive_chunk:
        mode = "AGGREGATE"

    pool = Pool(processes=12)
    results_for_aggregation = []
    with open(infile, "r") as f:
        while True:
            lines = f.readlines(file_chunk_size)
            bytes_read = sum(map(sys.getsizeof, lines))
            bytes_processed += bytes_read
            # TODO remove
            # print(f"Read {format_size(bytes_read)} from file")

            if len(lines) == 0:
                break

            line_chunk_size = len(lines) // n_futures
            line_chunks = chunk_list(lines, line_chunk_size)

            # futures = [mapper.remote(chunk) for chunk in line_chunks]

            results = pool.map(mapper, line_chunks)

            if mode == "STREAM":
                on_receive_chunk(results)
            elif mode == "AGGREGATE":
                results_for_aggregation.extend(results)

            del lines
            del results
            del line_chunks
            elapsed = time.time() - start_time
            print_iter_stats(bytes_processed, file_size, elapsed)
    print()
    cursor.show()
    if mode == "AGGREGATE":
        return results_for_aggregation


def print_iter_stats(bytes_processed, file_size, elapsed):
    throughput = bytes_processed / elapsed
    percentage_processed = bytes_processed / file_size * 100
    est_time_remaining = (file_size - bytes_processed) / throughput

    # fmt: off
    outputs = [
        f"Throughput: {format_size(throughput).ljust(10)} / sec",
        f"Processed: {format_size(bytes_processed).ljust(5)} / {format_size(file_size).ljust(5)} ({percentage_processed:.02f}%)",
        f"Est. time remaining: {format_timespan(est_time_remaining).ljust(5)}"

    ]

    # fmt: on

    print(
        ", ".join(outputs), end="\r",
    )
    sys.stdout.flush()
