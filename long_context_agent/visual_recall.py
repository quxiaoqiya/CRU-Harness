"""Optional local image-content recall; no new runtime dependency in the core."""
from dataclasses import dataclass
from .trajectory_memory_v2 import VisualStateIndex


def image_distance(a,b):
    bits=((a['horizontal']^b['horizontal']).bit_count()+(a['vertical']^b['vertical']).bit_count())/128
    color=sum(abs(x-y) for x,y in zip(a['colors'],b['colors']))/(192*255)
    return .55*bits+.45*color


class PerceptualVisualIndex(VisualStateIndex):
    """Combine local screenshot similarity, observed text, and transition links.

Similarity is a layout heuristic, not semantic equivalence or answer evidence.
The original image/state is still supplied to the bounded reader for checking.
"""
    def __init__(self,store,features,query_image,config=None):
        super().__init__(store,config)
        records={r['path']:r for r in features['features']}
        query=records[query_image]
        ranked=[]
        for i,state in enumerate(self.states):
            binding=state[0]
            if binding.image and binding.image in records:
                ranked.append((image_distance(query,records[binding.image]),i,records[binding.image]['image_sha256']))
        self.visual_neighbors=sorted(ranked,key=lambda x:(x[0],x[1]))[:24]
        self.visual_trace=[]

    def retrieve(self,queries,seen,*,temporal=False,preferred=()):
        baseline=super().retrieve(queries,seen,temporal=temporal,preferred=preferred)
        visuals=[];used=set(seen);trajectories={}
        for distance,i,sha in self.visual_neighbors:
            binding=self.states[i][0]
            if distance>.36 or trajectories.get(binding.trajectory_id,0)>=2:continue
            candidates=[i]
            if temporal:
                candidates += [self.keys.get((binding.trajectory_id,binding.state_index+d)) for d in (-1,1)]
            for j in candidates:
                if j is None or j in used:continue
                card=self.card(j,' '.join(queries),score=1-distance)
                visuals.append((j,card));used.add(j)
                self.visual_trace.append({'trajectory_id':card.trajectory_id,'state_index':card.state_index,
                    'distance':distance,'anchor_image_sha256':sha,'matched_state_index':binding.state_index,'anchor_neighbor':j==i})
                if len(visuals)>=self.config.max_cards//2:break
            trajectories[binding.trajectory_id]=trajectories.get(binding.trajectory_id,0)+1
            if len(visuals)>=self.config.max_cards//2:break
        result=visuals[:]
        for pair in baseline:
            if pair[0] not in {i for i,_ in result}:result.append(pair)
            if len(result)>=self.config.max_cards:break
        return result


def route_for_query(query_image):
    return 'perceptual_visual_guard' if query_image else 'legacy_trajectory_harness'
