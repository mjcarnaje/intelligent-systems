import os
import requests
import pandas as pd

api_key = "AIzaSyAspC8b-jnw9pWqfN3CS5j8vqoxiLCG1_w"
cse_id = "546362936983-ilolkq16ndchq9kc3qpskfu5sav1b3q9.apps.googleusercontent.com"

def search(search_item, api_key, cse_id, search_depth=10, site_filter=None):
    service_url = 'https://www.googleapis.com/customsearch/v1'

    params = {
        'q': search_item,
        'key': api_key,
        'cx': cse_id,
        'num': search_depth
    }

    try:
        response = requests.get(service_url, params=params)
        response.raise_for_status()
        results = response.json()

        if 'items' in results:
            if site_filter is not None:
                
                # Filter results to include only those with site_filter in the link
                filtered_results = [result for result in results['items'] if site_filter in result['link']]

                if filtered_results:
                    return filtered_results
                else:
                    print(f"No results with {site_filter} found.")
                    return []
            else:
                if 'items' in results:
                    return results['items']
                else:
                    print("No search results found.")
                    return []

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the search: {e}")
        return []

basketball_players_styles = pd.read_csv("data/basketball_players_styles.csv")

for player in basketball_players_styles["Player"].tolist():
    query = f"What is the style of {player}, it is Offensive, Defensive, or Balanced?"
    search_items = search(search_item=query, api_key=api_key, cse_id=cse_id, search_depth=10, site_filter=None)
    print(search_items)
    break # for testing purposes