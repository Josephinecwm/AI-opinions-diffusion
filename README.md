# Network Analytics - AI opinions diffusion on trading floor


The compound of technologies we use to call 'AI' promises to revolutionize many sectors. However, there is a substantial gap between what firms say they do with AI and what they actually do with it. In the context of knowledge intensive industries, there is yet another obstacle to the diffusion of AI, namely, 'people.' While some professionals may be thrilled to integrate AI in their daily work, some others may just feel threatened.

This project aims to help an investment bank to sustain the diffusion of positive AI opinions among traders on a specific trading floor using network analytics. Network analytics is the application of big data to analyse trend, functioning, and behaviour occuring on networks. The main tasks include identifying the network-related obstacles to the diffusion of positive AI opinions and implmenting recommendations to persuade traders to engage more with AI tools when it comes evaluating securities.

## Data information
trading_floor.xml provides information on the layout of the trading floor and the location of 192 traders (as reported by x-pos and y-pos).
The dataset also contains:
- a trader's opinion about the contribution of AI to his/her productivity and effectiveness in evaluating securities (1 = not at all; 10 = to a great extent), this variable is reported as the node attribute ai.
- the undirected network of knowledge exchange between traders (traders A and B are connected when A says he/she shares technical and industry knowledge with B and vice versa)
