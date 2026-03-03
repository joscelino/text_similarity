import time
import random

from text_similarity.api import Comparator

def run_benchmark():
    print("Iniciando benchmark do Comparator...")
    comp = Comparator.smart(entities=["product_model"], use_cache=True)
    
    query = "Notebook Dell Inspiron 15"
    base_candidates = [
        "Dell Inspiron 15 polegadas i5",
        "Notebook HP Pavilion",
        "Monitor Dell 24",
        "Notebook Lenovo Thinkpad",
        "Bateria para Dell Inspiron 15",
        "Teclado mecanico",
        "Mouse sem fio logitech",
        "Cadeira gamer corsair",
        "Macbook Pro 13 m1",
        "Placa de video rtx 3060"
    ]
    
    for size in [1000, 5000, 10000]:
        print(f"\nGenerando dataset de {size} items...")
        # Expandindo artificialmente 
        dataset = []
        for i in range(size):
            base = random.choice(base_candidates)
            dataset.append(f"{base} - ID{i}")
            
        comp.clear_cache()
        
        # Medindo 1-a-1
        print("Executando comparacao 1-a-1...")
        start_1to1 = time.time()
        results_1to1 = []
        for cand in dataset:
            score = comp.compare(query, cand)
            if score > 0.5:
                results_1to1.append((cand, score))
        results_1to1.sort(key=lambda x: x[1], reverse=True)
        time_1to1 = time.time() - start_1to1
        print(f"Tempo 1-a-1: {time_1to1:.4f}s")
        
        comp.clear_cache()
        
        # Medindo Batch
        print("Executando comparacao em Batch (compare_batch)...")
        start_batch = time.time()
        results_batch = comp.compare_batch(query, dataset, top_n=50, min_cosine=0.1)
        time_batch = time.time() - start_batch
        print(f"Tempo Batch: {time_batch:.4f}s")
        
        if time_batch > 0:
            speedup = time_1to1 / time_batch
            print(f"Speedup: {speedup:.2f}x mais rapido")

if __name__ == "__main__":
    run_benchmark()
