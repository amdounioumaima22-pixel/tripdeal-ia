import requests

BASE_URL = "http://localhost:5000"

# Test 1 : Health
r = requests.get(f"{BASE_URL}/health")
print("Health:", r.json())

# Test 2 : Simulation négociation complète
print("\n=== SIMULATION NEGOCIATION ===")
print("Client veut Paris, budget 5200 TND, prix affiche 6500 TND")
print("-" * 50)

prix_actuel = 6500
prix_plancher = 4800
budget = 5200
tour = 0

while tour < 6:
    tour += 1
    marge = prix_actuel - prix_plancher

    body = {
        "destination": "Paris, France",
        "prix_actuel": prix_actuel,
        "prix_plancher": prix_plancher,
        "budget_client": budget,
        "marge_actuelle": marge,
        "tour": tour
    }

    r = requests.post(f"{BASE_URL}/negotiate", json=body)
    data = r.json()

    print(f"\nTour {tour}:")
    print(f"  Action     : {data['action']}")
    print(f"  Prix avant : {data['prix_precedent']} TND")
    print(f"  Prix apres : {data['nouveau_prix']} TND")
    print(f"  Reduction  : {data['reduction']} TND")
    print(f"  Message    : {data['message']}")
    print(f"  Deal possible : {data['deal_possible']}")

    prix_actuel = data['nouveau_prix']

    if data['deal_possible']:
        print(f"\n DEAL CONCLU a {prix_actuel} TND !")
        break

    if data['action'] == 'refuser_negociation':
        print(f"\n Negociation echouee - Prix plancher atteint")
        break