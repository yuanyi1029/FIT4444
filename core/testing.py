# from core.inference import * 
# from config import * 

# def test_model(model): 
#     dataset_path = "dataset_hitl/labeled"
#     metrics = model.val(
#         data=dataset_path, 
#         split="test", 
#         plots=False, 
#         seed=SEED, 
#         deterministic=True,
#         # device=GPU_ID,
#         device='cpu'
#     )
    
#     return metrics.results_dict

# if __name__ == "__main__": 
#     # one, two = load_model("models/bests.pt")
#     one, two = load_model("models/finetuned/bests_v1.pt")
#     dct = test_model(one)
#     print(dct)