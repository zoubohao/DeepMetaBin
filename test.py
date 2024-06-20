from embed_seqs import embedding_contigs


if __name__ == "__main__":
    inputs_contigs = {"contig_1": "ATTTCCGGAAA"}
    output = embedding_contigs(inputs_contigs, "cuda:0")
    print(output)
    # {'contig_1': array([ 0.03165109,  0.04535764,  0.00375956, ..., -0.03616578, -0.02415844, -0.03955143], dtype=float32)}