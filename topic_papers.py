import pandas as pd
import numpy as np
import re
import tqdm
import matplotlib.pyplot as plt
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.nmf import Nmf
from gensim.models.coherencemodel import CoherenceModel
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import argparse
import sys
import pdb

TOKENIZERS_PARALLELISM=False

def clean_text(df, text_field):
    df[text_field] = df[text_field].apply(lambda elem: str(elem).replace('\n', ' '))
    df[text_field] = df[text_field].apply(lambda elem: str(elem).replace('  ', ' '))
    return df


def legacy_topic_modeling(df, col):
    f = open('stopwords', 'r')
    stopwords = set(eval(f.read()))

    seed = np.random.seed(1987)
    # lower case all words
    df = clean_text(df, col)
    # frequency calculations
    min_df = 1
    max_df = 1
    # tokenize documents
    par = [[w for w in re.findall(r'\b\w\w+\b' , p.lower()) if w not in stopwords] for p in df[col]]
    d_p = Dictionary(par)
    d_p.filter_extremes(no_below=min_df, no_above=max_df)
    # bag of words
    bow = [d_p.doc2bow(p) for p in par]
    # # tf idf transformation
    # tfidf = TfidfModel(bow)
    # X = tfidf[bow]
    # # Non negative matrix factorization
    # nmf = Nmf(X, num_topics=n_comp, id2word=d_p, kappa=0.1, eval_every=5)
    # nmf_coherence = CoherenceModel(model=nmf, texts=par, dictionary=d_p, coherence='c_v')
    # nmf_score = nmf_coherence.get_coherence()
    # print('---------------------')
    # print('NMF:')
    # print(nmf.show_topics(formatted=True, num_topics=n_comp))
    # print('nmf_score:{}'.format(nmf_score))
    #
    # # latent dirichlet allocation
    # lda = LdaModel(corpus=bow,
    #                id2word=d_p,
    #                chunksize=2000,
    #                alpha='auto',
    #                eta='auto',
    #                iterations=400,
    #                num_topics=n_comp,
    #                passes=20,
    #                eval_every=None,
    #                random_state=seed)
    # lda_coherence = CoherenceModel(model=lda, texts=par, dictionary=d_p, coherence='c_v')
    # lda_score = lda_coherence.get_coherence()
    # print('---------------------')
    # print('LDA:')
    # print(lda.show_topics(formatted=True, num_topics=n_comp))
    # print('lda_score:{}'.format(lda_score))

    # optimal topic calculations
    lda_para_model_n = []
    for n in tqdm.tqdm(range(2, 300, 5)):
        lda_model = LdaMulticore(corpus=bow,
                                 id2word=d_p,
                                 chunksize=2000,
                                 eta='auto',
                                 iterations=400,
                                 num_topics=n,
                                 passes=20,
                                 eval_every=None,
                                 random_state=seed)
        lda_coherence = CoherenceModel(model=lda_model,
                                       texts=par,
                                       dictionary=d_p,
                                       coherence='c_v')
        lda_para_model_n.append((n, lda_model, lda_coherence.get_coherence()))

    coh = pd.DataFrame(lda_para_model_n, columns=["n", "model", "coherence"]).set_index("n")[["coherence"]]
    print(coh.iloc[np.argmax(coh)])
    coh.plot(figsize=(16,9))
    plt.savefig('assets/coherence_{}.png'.format(col))
    plt.show()

def plot_embeddings(df, emb, col, dim=2):
    red = 'tsne'

    if red == 'pca':
        X_emb = PCA(n_components=dim,
                    ).fit_transform(emb)
    elif red == 'tsne':
        X_emb = TSNE(n_components=dim,
                     learning_rate='auto',
                     init='random',
                     perplexity=10).fit_transform(emb)

    if dim == 2:
        df['dim_1'] = X_emb[:, 0]
        df['dim_2'] = X_emb[:, 1]
        fig = px.scatter(df,
                         x='dim_1',
                         y='dim_2',
                         color='Year',
                         hover_name='Title',
                         hover_data=['Authors'])
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()

    elif dim == 3:
        df['dim_1'] = X_emb[:, 0]
        df['dim_2'] = X_emb[:, 1]
        df['dim_3'] = X_emb[:, 2]
        fig = px.scatter_3d(df,
                            x='dim_1',
                            y='dim_2',
                            z='dim_3',
                            color='Year',
                            hover_name='Title',
                            hover_data=['Authors'])
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()
    fig.write_html('assets/embeddings_{}_{}_{}d.html'.format(col, red, dim))
    pdb.set_trace()

def calculate_embeddings(df, col):
    # load model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # calculate sentence embeddings
    embeddings = model.encode(df[col])
    plot_embeddings(df, embeddings, col, dim=3)

    pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--type',
                        help='Select data to do nlp on [Title/Abstract]',
                        required=True)
    args = parser.parse_args()

    if args.type != 'Title' and args.type != 'Abstract':
        print('Choose valid type [Title/Abstract]!')
        sys.exit()
    df = pd.read_csv('data/scopus-16.csv')

    # legacy_topic_modeling(df, args.type)

    calculate_embeddings(df, args.type)
    pdb.set_trace()