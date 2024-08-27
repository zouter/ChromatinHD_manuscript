import torch
import gc

data = []
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
            data.append(
                {
                    "type": type(obj),
                    "size": obj.size(),
                    "device": str(obj.device),
                    "n": obj.numel(),
                    "dtype": str(obj.dtype),
                }
            )
    except:
        pass
data = pd.DataFrame(data)


###

# %%
import cProfile

stats = cProfile.run("loader.load(minibatch)", "restats")
import pstats

p = pstats.Stats("restats")
p.sort_stats("cumulative").print_stats()
