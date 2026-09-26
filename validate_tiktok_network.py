import pandas as pd
import networkx as nx
from collections import Counter

# Keep the CSV in the same VS Code folder as this script.
df = pd.read_csv('social_media_viral_content_dataset.csv')
tiktok = df[df['platform'] == 'TikTok'].copy()

# Validate the input before constructing the network.
required = ['post_id', 'hashtags', 'topic', 'is_viral', 'views', 'likes', 'comments', 'shares']
assert not tiktok[required].isna().any().any(), 'Missing required data'
assert tiktok['post_id'].is_unique, 'Duplicate post IDs'
assert set(tiktok['is_viral'].unique()).issubset({0, 1}), 'Unexpected viral labels'
assert (tiktok[['views', 'likes', 'comments', 'shares']] >= 0).all().all(), 'Negative engagement'

# Each post and each hashtag gets its own type of node.
G = nx.Graph()
raw_hashtag_counts = Counter()

for row in tiktok.itertuples(index=False):
    post = f'post:{row.post_id}'
    G.add_node(post, kind='post')
    tags = {tag.lower() for tag in row.hashtags.split() if tag.startswith('#')}
    for tag in tags:
        hashtag = f'tag:{tag}'
        G.add_node(hashtag, kind='hashtag')
        G.add_edge(post, hashtag)
        raw_hashtag_counts[tag] += 1

# Validate the graph against the input records.
assert sum(attrs['kind'] == 'post' for _, attrs in G.nodes(data=True)) == len(tiktok)
assert nx.number_of_selfloops(G) == 0, 'An unexpected self-loop was created'
assert all(G.nodes[u]['kind'] != G.nodes[v]['kind'] for u, v in G.edges()), 'Invalid edge type'
assert G.number_of_edges() == sum(raw_hashtag_counts.values())
assert all(G.degree(f'tag:{tag}') == count for tag, count in raw_hashtag_counts.items()), 'Hashtag counts disagree'

# Degree centrality follows the definition of importance used in the article.
degree_centrality = nx.degree_centrality(G)
pagerank = nx.pagerank(G)
print(f'Checks passed: {len(tiktok)} posts, {len(raw_hashtag_counts)} hashtags, {G.number_of_edges()} edges')
print('Top hashtags by degree:')
for tag, count in sorted(raw_hashtag_counts.items(), key=lambda item: (-item[1], item[0]))[:3]:
    print(f'  {tag}: {count} posts; degree centrality = {degree_centrality[f"tag:{tag}"]:.4f}')
print('Top hashtags by PageRank (a second network perspective):')
for tag in sorted(raw_hashtag_counts, key=lambda tag: -pagerank[f'tag:{tag}'])[:3]:
    print(f'  {tag}: {pagerank[f"tag:{tag}"]:.5f}')
