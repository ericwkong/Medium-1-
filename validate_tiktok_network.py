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


# Generate a table of the top 3 hashtags
import matplotlib.pyplot as plt

# Sort hashtags by number of connected posts
top_three = sorted(
    raw_hashtag_counts,
    key=lambda tag: (-raw_hashtag_counts[tag], -pagerank[f'tag:{tag}'])
)[:3]

# Prepare the table data
table_data = []

for tag in top_three:
    table_data.append({
        'Hashtag': tag,
        'Connected posts': raw_hashtag_counts[tag],
        'Degree centrality': f'{degree_centrality[f"tag:{tag}"]:.4f}',
        'PageRank': f'{pagerank[f"tag:{tag}"]:.5f}'
    })

table_df = pd.DataFrame(table_data)

# Print the table in the terminal
print(table_df.to_string(index=False))

# Save the table as a CSV
table_df.to_csv('tiktok_top_hashtags.csv', index=False)

# Create an image of the table
fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis('off')

table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    cellLoc='center',
    colLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.6)

# Make column headings bold
for col in range(len(table_df.columns)):
    table[(0, col)].set_text_props(weight='bold')

ax.set_title(
    'Most Important Hashtags in the TikTok Network',
    fontsize=14,
    fontweight='bold',
    pad=15
)

plt.savefig('tiktok_top_hashtags.png', dpi=300, bbox_inches='tight')
plt.show()

