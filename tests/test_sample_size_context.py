import asyncio
import threading

import numpy as np
import pandas as pd
import pytest

import flasheda as fe


def test_block_sets_size_and_restores_it():
    before = fe.get_sample_size()
    with fe.sample_size(1_000):
        assert fe.get_sample_size() == 1_000
        with fe.sample_size(200):
            assert fe.get_sample_size() == 200
        assert fe.get_sample_size() == 1_000
    assert fe.get_sample_size() == before
    with pytest.raises(ValueError):
        with fe.sample_size(10):
            pass


def test_analyze_uses_the_block_size():
    df = pd.DataFrame({"x": np.arange(3_000)})
    with fe.sample_size(500):
        assert fe.analyze(df, save_pdf=False).sample_size == 500


def test_threads_do_not_see_each_other():
    seen = {}
    start = threading.Barrier(2)

    def worker(n):
        with fe.sample_size(n):
            start.wait()                     # both blocks are open at the same time
            seen[n] = fe.get_sample_size()

    threads = [threading.Thread(target=worker, args=(n,)) for n in (300, 700)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {300: 300, 700: 700}


def test_async_tasks_do_not_see_each_other():
    async def task(n):
        with fe.sample_size(n):
            await asyncio.sleep(0.01)
            return fe.get_sample_size()

    async def main():
        return await asyncio.gather(task(400), task(900))

    assert asyncio.run(main()) == [400, 900]
