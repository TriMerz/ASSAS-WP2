#!/usr/bin/env python3

import gv_model as gvm
import logging
import tqdm
import sys
import glob
import os
import pandas as pd
# import more_itertools
from typing import Iterable, Tuple

import pyastec as pyas
sys.path.append(os.path.join('..','fluent'))
import fluent
import fluent.cache as cache
import fluent.path as pa
import fluent.pandas_util

def generate_folders_and_keys(dir_pattern:str):
    for dir in glob.glob(dir_pattern, recursive=True):
        yield dir, f"...{dir[-30:]}".replace("/", "_")

def generate_paths_and_values(base:cache.CachedOptOdbase, paths:Iterable[pa.Path]):
    for path in paths:
        if path.exists_from(base):
            yield (path, path.get_from(base))

def generate_folder_and_times(folder:str) -> Iterable[Tuple[str, float]]:
    times = pyas.tools.get_list_of_saving_time_from_path(folder)
    for time in times:
        yield folder, time

def generate_folders_times_bases(folders_and_times:Iterable[Tuple[str, float]], paths:Iterable[pa.Path]):
    for (folder, time) in folders_and_times:
        base = pyas.odloaddir(folder, time)
        cached_base = cache.cached(base)
        yield folder, time, generate_paths_and_values(cached_base, paths)
        pyas.odbase_delete(base)

def chunked(times:Iterable[Tuple[str, float]], size:int):

    result=list()
    n=0
    tmp=list()
    for i in times:
        tmp.append(i)
        n=n+1
        if( n==size ):
            result.append(tmp)
            tmp=list()
            n=0
    if( n>0 ):
        result.append(tmp)

    return result

def extract(dir_pattern:str, output_folder:str, output_file:str, batch_size:int):

    folders_and_keys = list(generate_folders_and_keys(dir_pattern))

    initial_base = pyas.odloaddir(folders_and_keys[0][0], 0.)

    gv_model_metadata = gvm.get_gv_model_metadata(initial_base)

    gvm.dump_gv_model_metadata(os.path.join(output_folder, gvm.METADATA_FILE), gv_model_metadata)

    paths = list(gvm.generate_paths(gv_model_metadata.name_mapping))
    gv_to_path_name_to_common_name = dict(gvm.gv_to_path_name_to_gv_name(gv_model_metadata.name_mapping))

    logging.info(f"Generated {len(paths)} paths, saving them")

    for (folder, key) in folders_and_keys:

        folders_times = generate_folder_and_times(folder)

        folders_and_times_batched = chunked(folders_times, batch_size)
    
        for (index, folders_times) in enumerate(folders_and_times_batched):
            logging.debug(f"Starting batch {index + 1} of folder {key}")

            count_bases = len(folders_times)

            bar_folders_and_times = tqdm.tqdm(folders_times)
            bar_folders_and_times.set_postfix_str(key)
            dataset = generate_folders_times_bases(bar_folders_and_times, paths)

            df:pd.DataFrame = fluent.pandas_util.to_df(dataset, count_bases)

            for gv in gvm.GVS:
                path_name_to_common_name = gv_to_path_name_to_common_name[gv]
                gv_path_names = list(path_name_to_common_name.keys())
                gv_df = df[gv_path_names]
                renamed_gv_df = gv_df.rename(path_name_to_common_name, axis=1)

                output_batch_file_name = f"{output_folder}/{output_file}-{key}-gv{gv}-{index}.pkl.gz"
                logging.debug(f"Saving batch to file {output_batch_file_name}")
                renamed_gv_df.to_pickle(output_batch_file_name, compression="gzip")

if __name__ == "__main__":

    dir_pattern = sys.argv[1]
    output_folder = sys.argv[2]
    output_file_prefix = sys.argv[3]
    batch_size = int(sys.argv[4])

    extract(dir_pattern, output_folder, output_file_prefix, batch_size)
