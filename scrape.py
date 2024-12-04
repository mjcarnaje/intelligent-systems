import requests
from bs4 import BeautifulSoup
import csv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the base URL and the years to scrape
base_url = "https://basketball.realgm.com/nba/allstar/game/rosters/"
years = range(2011, 2025)

# Open a CSV file to write the data
with open('data/all_stars.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Year', 'Player'])  # Write the header row

    # Iterate through each year
    for year in years:
        url = f"{base_url}{year}"
        logging.info(f"Fetching data for year: {year}")
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Find the table containing the roster
            table = soup.find('table', {'class': 'basketball compact'})
            if table:
                logging.info(f"Roster table found for {year}.")
                # Iterate through each row in the table body
                for row in table.tbody.find_all('tr'):
                    # Find the cell containing the player's name
                    player_cell = row.find('td', {'data-th': 'Player'})
                    if player_cell:
                        player_name = player_cell.get_text(strip=True)

                        if ', Jr.' in player_name:
                            player_name = player_name.replace(', Jr.', ' Jr.')

                        if 'Yao Ming' in player_name:
                            continue
                        
                        writer.writerow([f"{year-1}-{str(year)[2:]}", player_name])
                        logging.info(f"Added player: {player_name} for year: {year}")
            else:
                logging.warning(f"No roster table found for {year}.")
        else:
            logging.error(f"Failed to retrieve data for {year}. Status code: {response.status_code}")
