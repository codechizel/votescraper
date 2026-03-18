# Chapter 3: Co-Voting Networks: Who Votes Together?

> *Clustering treats each legislator as a point on a line. Network analysis treats each legislator as a node in a web. The line tells you where people are. The web tells you who's connected to whom — and who sits at the crossroads.*

---

## Why Networks?

Chapters 1 and 2 asked "are there groups?" and found two (the parties). But groups are only part of the story. Within each group, some legislators are more central than others. Between the groups, a few legislators act as bridges — they vote with their own party most of the time but cross the aisle often enough to connect the two sides.

These roles — broker, bridge-builder, hub, peripheral backbencher — are invisible to clustering. A legislator in the middle of the Republican distribution looks moderate in a silhouette plot, but you can't tell whether they're a quiet moderate who rarely crosses the aisle or a noisy centrist who collaborates with Democrats regularly.

**Network analysis** reveals these roles by representing the legislature as a **graph**: nodes (legislators) connected by edges (co-voting relationships). The shape of the graph — who's connected to whom, how strongly, and through how many intermediaries — tells you things that ideology scores alone cannot.

## Building the Network

### Nodes and Edges

Every legislator who passes the participation filter (at least 20 contested votes) becomes a **node**. Every pair of legislators gets an **edge** weighted by their Cohen's Kappa agreement score.

But not every edge is worth drawing. Two legislators with Kappa = 0.05 barely agree more than chance — that's not a meaningful connection. Tallgrass applies a **threshold**: only pairs with Kappa above a cutoff become edges in the network. The default cutoff is 0.40, which corresponds to the Landis-Koch threshold for "moderate" or better agreement.

This creates a sparse network where only meaningful connections survive. In the typical Kansas House, the threshold keeps around 50–60% of all possible edges.

### Why Kappa, Not Raw Agreement?

The same logic from Volume 3 applies. Two random legislators agree 70% of the time in Kansas because of the 82% Yea base rate. A network built on raw agreement would connect almost everyone to everyone — a useless hairball. Kappa strips out the chance agreement, so edges represent genuine co-voting relationships that go beyond what you'd expect by accident.

### The Distance Trick

Some centrality measures need *distances* rather than *similarities* — the shorter the path, the closer the connection. Tallgrass converts similarities to distances by taking the reciprocal:

```
distance = 1 / Kappa
```

High Kappa (strong agreement) becomes a short distance. Low Kappa becomes a long distance. This ensures that path-based measures like betweenness and closeness work correctly — the "shortest path" between two legislators passes through their strongest allies, not their weakest connections.

**Codebase:** `analysis/11_network/network.py` (`build_kappa_network()`, `KAPPA_THRESHOLD_DEFAULT = 0.40`)

## Centrality: Who Matters?

In a social network, not all nodes are equally important. Some people have many connections. Some sit at the intersection of different groups. Some are connected to other highly connected people. **Centrality measures** formalize these intuitions.

Tallgrass computes seven centrality measures for every legislator. Here's what each one tells you, explained through a workplace analogy:

### Degree Centrality: The Socialite

**Plain English:** "What fraction of other legislators do you have a meaningful co-voting relationship with?"

If a legislator is connected to 80 out of 120 possible colleagues, their degree centrality is 80/120 = 0.67. This is the simplest centrality measure — just count the connections.

In a workplace, this is the person who knows everyone in the building. They may not be the most influential, but they're in a lot of conversations.

### Weighted Degree: The Deep Connector

**Plain English:** "What's the total strength of all your connections?"

While degree just counts connections, weighted degree sums the Kappa values of all edges. A legislator with 50 connections at Kappa = 0.6 has a higher weighted degree than one with 80 connections at Kappa = 0.3.

In a workplace, this is the person who doesn't just know many people but has *strong* relationships with them — the difference between having 500 LinkedIn connections and having 50 close colleagues you'd call at midnight.

### Betweenness Centrality: The Broker

**Plain English:** "How often do you sit on the shortest path between two other legislators?"

This is the most politically interesting centrality measure. Betweenness identifies legislators who **connect otherwise disconnected groups**. A moderate Republican with high betweenness is the legislator through whom information, influence, or coalition-building flows between the conservative wing and the Democratic caucus.

The calculation: for every pair of legislators (A, B), find the shortest path between them in the network. If legislator C lies on that path, C gets credit. Betweenness is the fraction of all shortest paths that pass through a given legislator.

In a workplace, this is the person who connects the engineering team to the sales team because they have relationships in both. Remove this person and the two sides lose contact.

**The formula:**

```
Betweenness(v) = sum over all pairs (s,t) of:
    (number of shortest paths from s to t that pass through v)
    ÷ (total number of shortest paths from s to t)
```

### Eigenvector Centrality: The Well-Connected

**Plain English:** "Are you connected to other well-connected legislators?"

Degree treats all connections equally. Eigenvector centrality gives extra credit for being connected to *important* nodes. A legislator connected to two party leaders has higher eigenvector centrality than one connected to two backbenchers, even if both have the same degree.

The name comes from the mathematical technique (eigenvector decomposition of the adjacency matrix), but the intuition is simple: your importance depends on the importance of your neighbors, which depends on the importance of *their* neighbors, and so on. The calculation resolves this circular definition through iteration.

In a workplace, this is the person who may not know everyone, but they know all the decision-makers.

### Closeness Centrality: The Efficient Communicator

**Plain English:** "How quickly can you reach everyone else in the network?"

Closeness measures the average shortest-path distance from a legislator to all other legislators. A legislator at the center of the network has short paths to everyone; a peripheral legislator has long paths.

In a workplace, this is the person who can get a message to anyone in the organization with at most one or two intermediaries.

### PageRank: The Random Walker's Favorite

**Plain English:** "If you randomly followed voting connections, how often would you end up at this legislator?"

PageRank was invented by Larry Page and Sergey Brin for ranking web pages (hence Google), but it works on any network. It simulates a random walk: start at a random legislator, follow a random connection, repeat. The legislators you visit most often have the highest PageRank.

The difference from eigenvector centrality is subtle: PageRank accounts for the *number* of connections each neighbor has. Being connected to a well-connected legislator is less impressive if that legislator is connected to 100 others (your share of their attention is thin).

### Cross-Party Fraction: The Bridge-Builder

**Plain English:** "What fraction of your connections are to legislators from the other party?"

This isn't a standard centrality measure — it's a Tallgrass-specific diagnostic. A legislator with 60 connections, 5 of which are to the other party, has a cross-party fraction of 5/60 = 0.08. One with 40 connections and 12 cross-party has 12/40 = 0.30.

Legislators with high cross-party fraction and high betweenness are the true bridge-builders — they don't just sit near the center of their own party, they actively maintain voting relationships across the aisle.

**Codebase:** `analysis/11_network/network.py` (`compute_centralities()`, `identify_bridge_legislators()`, `TOP_BRIDGE_N = 15`)

## Community Detection: Finding Structure in the Web

### The Leiden Algorithm

Centrality tells you about individual nodes. **Community detection** finds groups of nodes that are more densely connected to each other than to the rest of the network.

Tallgrass uses the **Leiden algorithm** (Traag, Waltman, and van Eck, 2019), which was designed to fix a fundamental flaw in the widely used Louvain algorithm.

**The Louvain problem:** The Louvain algorithm (Blondel et al., 2008) finds communities by optimizing a measure called **modularity** — the difference between the density of connections within communities and the density you'd expect by chance. But Louvain can produce communities that are **internally disconnected**: it assigns two legislators to the same community even though there's no path of strong connections between them within that community. In experiments, up to 25% of Louvain communities had this problem.

**The Leiden fix:** Leiden adds a refinement step between each round of optimization. After grouping nodes, it checks whether each community is internally connected. If a community has disconnected pieces, it splits them. The result is guaranteed to be well-connected — every community is a true cluster of mutually reachable nodes.

### Modularity and Resolution

Modularity measures how much the network departs from random connectivity:

```
Q = (fraction of edges within communities) − (expected fraction if edges were random)
```

**Plain English:** "Are same-community legislators connected more than chance would predict?"

High modularity (Q > 0.3) means the network has strong community structure. Low modularity means the network is relatively homogeneous.

But modularity has a resolution parameter that controls how fine-grained the communities are. At low resolution, the algorithm finds two big communities (the parties). At high resolution, it carves the legislature into many small cliques.

Tallgrass sweeps across 8 resolution values (0.5 to 3.0) to see how the community structure changes. The stable pattern — one that persists across multiple resolutions — is more trustworthy than a structure that appears at only one setting.

### CPM: Resolution-Limit-Free Detection

Standard modularity optimization has a known weakness: it can't detect communities smaller than a certain size that depends on the total network (the "resolution limit"). Tallgrass also runs the **Constant Potts Model** (CPM) variant of Leiden, which doesn't have this limit. CPM can find small subcaucuses — a clique of 5 moderate Republicans who consistently vote together — even in a network of 125 legislators.

**Codebase:** `analysis/11_network/network.py` (`detect_communities_multi_resolution()`, `detect_communities_cpm()`, `LEIDEN_RESOLUTIONS`, `CPM_GAMMAS`)

## Threshold Sensitivity

The Kappa threshold of 0.40 is a judgment call. What happens if we raise or lower it?

Tallgrass runs a **threshold sweep** at 0.30, 0.40, 0.50, and 0.60:

| Threshold | What happens |
|-----------|-------------|
| 0.30 (loose) | Dense network — most within-party pairs connected, some cross-party edges |
| 0.40 (default) | Balanced — clear party structure, bridge legislators visible |
| 0.50 (strict) | Sparse — only strong within-party allies connected |
| 0.60 (very strict) | Very sparse — only the tightest cliques survive |

At all thresholds, the two-party structure dominates. The community detection finds two communities matching the parties at every threshold. What changes is the density within each community and the number of cross-party bridges.

The consistency across thresholds is itself a finding: the party structure isn't an artifact of any particular cutoff. It's robust.

**Codebase:** `analysis/11_network/network.py` (`run_threshold_sweep()`, `KAPPA_THRESHOLD_SENSITIVITY`)

## Backbone Extraction: The Skeleton of the Network

Even at the default threshold, the network can be dense enough that visualizations become hard to read. **Backbone extraction** strips the network down to its most important edges, like an X-ray showing only the skeleton.

Tallgrass uses the **disparity filter** (Serrano, Boguna, and Vespignani, 2009). For each legislator, it asks: "If this legislator's total connection strength were distributed randomly across all their edges, would any individual edge be surprisingly strong?"

The null hypothesis is that edge weights are uniformly distributed across a node's connections. An edge that's significantly stronger than the null prediction (p < 0.05) is retained. Everything else is filtered out.

The beauty of this approach is that it's **multiscale**: it preserves important edges for both highly connected hubs (party leaders with many strong alliances) and peripheral legislators (who might have only 3 connections, but one of them is anomalously strong). A simpler approach — like "keep only edges above Kappa = 0.60" — would erase peripheral structure entirely while keeping too many hub edges.

**Codebase:** `analysis/11_network/network.py` (`disparity_filter()`)

## Subnetwork Analysis

### Within-Party Networks

Tallgrass builds separate networks for Republicans and Democrats, then runs Leiden community detection within each. This reveals **intra-party factions** that are invisible in the full network.

In the Republican caucus, Leiden sometimes finds two or three sub-communities at higher resolutions — but these communities have fuzzy boundaries and swap members easily across resolutions, consistent with the continuous gradient from Chapter 1.

### High-Discrimination Networks

Instead of using all votes, this subnetwork uses only **high-discrimination bills** — those with |β_mean| > 1.5 from IRT. These are the bills that most sharply separate ideologies. The resulting network reveals voting relationships on the votes that *matter most* for ideological positioning, stripping out the noise from procedural and unanimous votes.

**Codebase:** `analysis/11_network/network.py` (`build_within_party_network()`, `build_high_disc_network()`, `HIGH_DISC_THRESHOLD = 1.5`)

## What the Network Reveals About Kansas

### Party Modularity

The **party modularity** — the modularity score when communities are defined by party labels rather than by the algorithm — is a single-number summary of polarization. It measures how much the party structure explains the network's connectivity.

In Kansas, party modularity typically ranges from 0.30 to 0.45 — strong structure. This confirms that the dominant organizing principle of co-voting is party membership, just as clustering and LCA found.

### The Bridge Legislators

The most politically interesting output is the **bridge legislator** table: the top 15 legislators ranked by betweenness centrality, annotated with their cross-party fraction. These are the people who hold the network together across the partisan divide.

In a supermajority chamber like Kansas (75%+ Republican), bridge legislators are often moderate Republicans — they vote with their party on most issues but maintain enough cross-party connections to appear on shortest paths between the two communities. Their high betweenness makes them disproportionately important for cross-party legislation.

---

## Key Takeaway

Network analysis reveals what ideology scores miss: the *relational structure* of the legislature. Centrality measures identify who occupies key positions — the hubs, the brokers, the bridge-builders. Community detection confirms the two-party structure while revealing how resolution affects the picture. The disparity filter extracts the skeleton of the network, showing the strongest and most unexpected connections. Together, these tools paint a portrait of the legislature as a social system, not just a collection of individual scores.

---

*Terms introduced: network (graph), node, edge, Kappa threshold, degree centrality, weighted degree, betweenness centrality, eigenvector centrality, closeness centrality, PageRank, cross-party fraction, Leiden algorithm, modularity, resolution parameter, CPM (Constant Potts Model), threshold sweep, disparity filter, backbone extraction, party modularity*

*Next: [Bipartite Networks: Bills That Bridge the Aisle](ch04-bipartite-networks.md)*
