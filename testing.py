from inference import * 

def test_model(model): 
    dataset_path = "C:/Users/wyuan/Projects/FIT4444/dataset_hitl/labeled"
    metrics = model.val(data=dataset_path, split="test", plots=False, seed=42, deterministic=True)
    
    return metrics.results_dict

if __name__ == "__main__": 
    one, two = load_model("models/bests.pt")
    dct = test_model(one)
    print(dct)