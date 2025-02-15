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
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
# import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import argparse
import sys
import os
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

def plot_embeddings(df, emb, col, dim):
    red = 'tsne'
    if os.path.exists('data/output_{}_{}_{}d_emb.csv'.format(col, red, dim)):
        df = pd.read_csv('data/output_{}_{}_{}d_emb.csv'.format(col, red, dim), sep=';')
        return df
    else:
        seed = np.random.seed(1987)
        plot_flag = True
        pca = PCA(whiten=False, random_state=seed)
        data_pca = pca.fit_transform(emb)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        n_comp = np.argmax(cum_var >= 0.90) + 1
        print(f'Keeping {n_comp}/{emb.shape[1]} components to retain {np.sum(pca.explained_variance_ratio_[:n_comp]) * 100}% of explained variance!')
        
        data_pca = data_pca[:, :n_comp]

        if red == 'tsne':
            X_emb = TSNE(n_components=dim,
                        learning_rate='auto',
                        init='pca',
                        random_state=seed,
                        n_jobs=-1,
                        perplexity=20).fit_transform(data_pca)

        if dim == 2:
            df['dim_1'] = X_emb[:, 0]
            df['dim_2'] = X_emb[:, 1]
        elif dim == 3:
            df['dim_1'] = X_emb[:, 0]
            df['dim_2'] = X_emb[:, 1]
            df['dim_3'] = X_emb[:, 2]

        if plot_flag:
            if dim == 2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                                x=df['dim_1'],
                                y=df['dim_2'],
                                mode='markers',
                                marker_color=df['Year'],
                                text=df['Title'],
                                marker=dict(
                                    size=8,
                                    showscale=True
                                )))
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                fig.show()

            elif dim == 3:
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                                x=df['dim_1'],
                                y=df['dim_2'],
                                z=df['dim_3'],
                                mode='markers',
                                marker_color=df['Year'],
                                text=df['Title'],
                                marker=dict(
                                    size=4,
                                    showscale=True
                                )))

                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
                fig.show()
            fig.write_html('assets/embeddings_{}_{}_{}d.html'.format(col, red, dim))

        df.to_csv('data/output_{}_{}_{}d_emb.csv'.format(col, red, dim), sep=';', index=False)
    return df

def calculate_embeddings(df, col, num_dim):
    # load model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # calculate sentence embeddings
    embeddings = model.encode(df[col])

    df = plot_embeddings(df, embeddings, col, num_dim)
    return df

def get_word_frequency_hdb(df, col):
    emb_cols = df.columns[df.columns.str.startswith('dim_')].tolist()
    # A low min_cluster_size leads to many clusters but also more noise.
    # A high min_cluster_size results in fewer clusters and more points marked as outliers (-1).
    # Look for a balance: fewer outliers while maintaining a reasonable number of clusters.  
    # selected min_sample_size=20 or 30 according to this
    test_outliers = True
    if test_outliers:
        X = df[emb_cols].values

        min_cluster_sizes = range(5, 51, 5)  # Test values from 5 to 50 in steps of 5
        num_clusters = []
        num_outliers = []

        for min_size in min_cluster_sizes:
            clusterer = HDBSCAN(min_cluster_size=min_size, metric='euclidean')
            labels = clusterer.fit_predict(X)
            
            num_clusters.append(len(set(labels)) - (1 if -1 in labels else 0))  # Exclude outliers (-1)
            num_outliers.append(sum(labels == -1))  # Count number of outliers

        # Plot results
        fig, ax1 = plt.subplots()

        ax1.set_xlabel("min_cluster_size")
        ax1.set_ylabel("Number of Clusters", color="tab:blue")
        ax1.plot(min_cluster_sizes, num_clusters, marker='o', color="tab:blue", label="Clusters")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Number of Outliers", color="tab:red")
        ax2.plot(min_cluster_sizes, num_outliers, marker='x', color="tab:red", label="Outliers")

        fig.tight_layout()
        plt.title("Effect of min_cluster_size on Clustering")
        plt.show()

    custom_stopwords = {
        'music', 'audio', 'of', 'for', 'the', 'a', 'an', 'in', 'to', 'on', 
        'and', 'with', 'by', 'from', 'at', 'this', 'that', 'it', 'is', 'as', 
        'mir', 'using', 'musical', 'song', 'ismir', 'digital', 'retrieval', 'information',
        'retrieving', 'data', 'learning', 'pattern', 'feature', 'retrieval.', 'analysis',
        'automatic'
    }

    hdbscan = HDBSCAN(min_cluster_size=5, metric='euclidean')
    hdb_clusters = hdbscan.fit_predict(df[emb_cols])
    df['title_clusters'] = hdb_clusters

    hdb_word = {}
    for hdb in np.unique(hdb_clusters).tolist():
        these_titles = df['Title'][df['title_clusters'] == hdb].tolist()
        all_words = [word.lower() for phrase in these_titles for word in phrase.split() 
             if word.isalpha()]
        filtered_words = [word for word in all_words if word not in custom_stopwords]
        if filtered_words:  # Ensure there's at least one word left
            most_common_word = Counter(filtered_words).most_common(1)[0][0]
            if most_common_word not in hdb_word.values():
                hdb_word[hdb] = most_common_word
            else:
                hdb_word[hdb] = Counter(filtered_words).most_common(10)[1][0]
        else:
            hdb_word[hdb] = None

    df['title_words'] = df['title_clusters'].map(hdb_word)

    if len(emb_cols) == 2:
        fig = go.Figure()
        for word in df['title_words'].unique():
            # Filter the data for this specific word
            word_data = df[df['title_words'] == word]
            
            # Add a trace for this word
            fig.add_trace(go.Scatter(
                x=word_data['dim_1'],
                y=word_data['dim_2'],
                mode='markers',
                text=word_data['Title'],  # Hover text
                name=word,  # Label the trace with the word
                marker=dict(
                    size=8,
                )
            ))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()

    elif len(emb_cols) == 3:
        fig = go.Figure()
        for word in df['title_words'].unique():
            # Filter the data for this specific word
            word_data = df[df['title_words'] == word]
            
            # Add a trace for this word
            fig.add_trace(go.Scatter3d(
                x=word_data['dim_1'],
                y=word_data['dim_2'],
                z=word_data['dim_3'],
                mode='markers',
                text=word_data['Title'],  # Hover text
                name=word,  # Label the trace with the word
                marker=dict(
                    size=4,
                )
            ))
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()
    fig.write_html('assets/embeddings_{}_{}_{}d.html'.format(col, 'tsne', len(emb_cols)))
    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--type',
                        help='Select data to do nlp on [Title/Abstract]',
                        required=True)
    parser.add_argument('--num_dim',
                        help='Select number of dimensions for reduction',
                        type=int,
                        default=2)
    args = parser.parse_args()

    if args.type != 'Title' and args.type != 'Abstract':
        print('Choose valid type [Title/Abstract]!')
        sys.exit()
    df = pd.read_csv('data/summary_dataset.csv')
    df.Abstract = df.Abstract.fillna('')

    # legacy_topic_modeling(df, args.type)

    df = calculate_embeddings(df, args.type, args.num_dim)

    # get possible topics based on word frequency
    df = get_word_frequency_hdb(df, args.type)
    pdb.set_trace()