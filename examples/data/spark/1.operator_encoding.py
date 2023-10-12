# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description:
#
# Created at 12/23/22

import argparse
import multiprocessing as mp

from utils.common import BenchmarkUtils, PickleUtils
from utils.data.extractor import replace_symbols, remove_hash_suffix, df_convert_query2op, brief_clean, get_csvs

from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec, TfidfModel, Doc2Vec
from gensim.corpora import Dictionary
from gensim.models.doc2vec import TaggedDocument

import numpy as np

from utils.model.utils import get_str_hash
from sklearn.metrics.pairwise import cosine_similarity

class Args():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("-b", "--benchmark", type=str, default="TPCH")
        self.parser.add_argument("--scale-factor", type=int, default=100)
        self.parser.add_argument("--src-path-header", type=str, default="resources/dataset/tpch_100_query_traces")
        self.parser.add_argument("--cache-header", type=str, default="examples/data/spark/cache")
        self.parser.add_argument("--mode", type=str, default="w2v")
        self.parser.add_argument("--vec-size", type=int, default=32)
        self.parser.add_argument("--alpha", type=float, default=0.025)
        self.parser.add_argument("--epochs", type=int, default=50)
        self.parser.add_argument("--downsamples", type=float, default=1e-3)
        self.parser.add_argument("--dm", type=int, default=1)
        self.parser.add_argument("--window", type=int, default=3)

    def parse(self):
        return self.parser.parse_args()


def data_preparation(df, cache_header):
    try:
        cache = PickleUtils.load(cache_header, "enc_cache_meta.pkl")
        op_df = cache["op_df"]
        oid_dict = cache["oid_dict"]
        print(f"found cached unique_op_df, shape={op_df.shape}")
    except:
        df = df.reset_index(drop=True).set_index(["sql_struct_id", "sql_struct_svid"])
        op_df = df_convert_query2op(df)
        op_df = op_df.apply(replace_symbols).apply(remove_hash_suffix).apply(brief_clean)
        n_ori = op_df.shape[0]
        op_df = op_df.to_frame()
        op_df["sign"] = op_df.planDescription.apply(get_str_hash)
        op_df_unique = op_df.drop_duplicates()
        sign_list = op_df_unique.sign.tolist()
        sign2oid = {sign: ind for ind, sign in enumerate(sign_list)}
        op_df["oid"] = op_df.sign.apply(lambda x: sign2oid[x])
        oid_dict = {}
        sids = df.reset_index().sql_struct_id.unique()
        for sid in sids:
            df_ = op_df.loc[sid]
            assert df_.groupby(level=0).size().std() == 0
            oid_dict[sid] = df_.oid.values.reshape(df_.index.unique().size, -1)
        n_unique = op_df_unique.shape[0]
        op_df = op_df_unique
        PickleUtils.save({
            "oid_dict": oid_dict,
            "op_df": op_df_unique
        }, cache_header, "enc_cache_meta.pkl")
        print(f"get {n_unique} unique clean operator desc out of {n_ori} operators")
    return op_df, oid_dict


def get_operator_descs(op_df):
    """
    return op_df and op_df_cat with cat1=(operator_type), cat2=(struct_id, operator_type)
    """
    # operator's desc will be cleaned for "\n" and "\n\n" at `SqlStructBefore`.
    op_df_cat = op_df.planDescription.apply(lambda x: x.split()[0]).reset_index()
    op_df_cat.columns = ["sid", "svid", "cat1"]
    op_df_cat["cat2"] = op_df_cat.apply(lambda r: f"{r['sid']}-{r['cat1']}", axis=1)
    cat1 = sorted(op_df_cat.cat1.unique())
    cat2 = sorted(op_df_cat.cat2.unique())
    op_df_cat["cat1_index"] = op_df_cat["cat1"].replace(cat1, list(range(len(cat1))))
    op_df_cat["cat2_index"] = op_df_cat["cat2"].replace(cat2, list(range(len(cat2))))
    return op_df_cat

def self_eval(op_df, op_encs):
    op_df_cat = get_operator_descs(op_df)
    dists = cosine_similarity(op_encs, op_encs)
    closet_ids = dists.argsort()[-2, :]
    self_cat1 = op_df_cat.cat1_index.values
    closet_cat1 = op_df_cat.cat1_index.values[closet_ids]
    rate_cat1 = sum(self_cat1 == closet_cat1) / len(self_cat1)
    self_cat2 = op_df_cat.cat2_index.values
    closet_cat2 = op_df_cat.cat2_index.values[closet_ids]
    rate_cat2 = sum(self_cat2 == closet_cat2) / len(self_cat2)
    print(f"match rate: cat1 ({rate_cat1*100:.0f}%), cat2 ({rate_cat2 * 100:.0f}%)")



if __name__ == "__main__":
    args = Args().parse()
    print(args)
    bm = args.benchmark.lower()
    sf = args.scale_factor
    src_path_header = args.src_path_header
    cache_header = f"{args.cache_header}/{bm}_{sf}"
    mode = args.mode
    assert mode in ("d2v", "w2v"), ValueError(mode)

    templates = [f"q{i}" for i in BenchmarkUtils.get(bm)]
    df = get_csvs(templates, src_path_header, cache_header, samplings=["lhs", "bo"])
    op_df, oid_dict = data_preparation(df, cache_header)

    if mode == "w2v":
        op_descs = [row.split() for row in op_df["planDescription"]]
        phrases = Phrases(op_descs, min_count=30)
        bigram = Phraser(phrases)
        op_descriptions = bigram[op_descs]
        w2v_model = Word2Vec(min_count=1,
                             window=args.window,  # 3
                             vector_size=args.vec_size,  # 32
                             sample=0.1,
                             alpha=args.alpha,  # 0.025
                             min_alpha=0.0007,
                             workers=mp.cpu_count() - 1)
        w2v_model.build_vocab(op_descriptions)
        w2v_model.train(op_descriptions, total_examples=w2v_model.corpus_count, epochs=args.epochs, report_delay=1)
        print(f"get {w2v_model.wv.vectors.shape[0]} words from word2vec")

        # https://stackoverflow.com/a/31738627
        dictionary = Dictionary()
        BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in op_descriptions]
        tfidf = TfidfModel(BoW_corpus, smartirs='ntc')
        print(f"get {len(tfidf.idfs)} words from tfidf")
        op_encs = np.array([np.mean([w2v_model.wv.get_vector(dictionary[wid], norm=True) * freq for wid, freq in wt], 0)
                            for wt in tfidf[BoW_corpus]])
    elif mode == "d2v":
        tokens_list = list(map(lambda x: x.split(), op_df.planDescription))
        corpus = [TaggedDocument(d, [i]) for i, d in enumerate(tokens_list)]
        d2v_model = Doc2Vec(min_count=1,
                            window=args.window,
                            vector_size=args.vec_size,
                            sample=0.1,
                            alpha=args.alpha,
                            min_alpha=0.0007,
                            workers=mp.cpu_count() - 1)
        d2v_model.build_vocab(corpus)
        d2v_model.train(corpus, total_examples=d2v_model.corpus_count, epochs=args.epochs)
        op_encs = d2v_model.dv.get_normed_vectors()
    else:
        raise ValueError(mode)

    PickleUtils.save({
        "oid_dict": oid_dict,
        "op_encs": op_encs
    }, cache_header, f"enc_cache_{mode}.pkl")

    self_eval(op_df, op_encs)