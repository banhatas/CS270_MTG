import pandas as pd
import mtgtools
import numpy as np
import pickle
import datetime

from mtgtools.MtgDB import MtgDB
from requests import get
from json import loads
from shutil import copyfileobj
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder

DB_PATH = '/home/tylerc/dat/school/mtg/'
DB_NAME = DB_PATH + 'mtg_db'
LAYOUT_IGNORE = ['token', 'art_series', 'double_faced_token', 'emblem', 'prototype', 'vanguard', 'reversible_card']
ATTRIBUTES = [
        'id',
        'name',
        # 'all_parts', 
        # 'card_faces', 
        'cmc', 
        'color_identity', 
        # 'color_indicator', 
        # 'colors', 
        'keywords', 
        'layout', 
        # 'loyalty', 
        # 'mana_cost', 
        'oracle_text', 
        # 'oversized', 
        # 'power', 
        'produced_mana', 
        'reserved', 
        # 'toughness', 
        'type_line', 
        'artist', 
        'booster', 
        'border_color', 
        # 'finishes', 
        'flavor_text', 
        'frame_effects', 
        'frame', 
        'full_art',
        # 'prices', 
        'promo',
        'rarity',
        'released_at', 
        'reprint', 
        'set', 
        # 'set_id', # duplicate of 'set_id' 
        'story_spotlight', 
        'textless', 
        'variation',
        'power_num', 
        'toughness_num', 
        # 'loyalty_num'
]

def update_db(time_tol = 3):
    '''
    Checks the last time the data base was updated and updates it.
    '''

    # open file and get days since last update
    upd_file = DB_PATH + 'update.txt'
    with open(upd_file, 'r') as f:
        date_str = f.read()
    last_upd = datetime.date(int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]))
    today = datetime.date.today()
    days_since_last_upd = (last_upd - today).days

    # if it has been a while, update database
    if days_since_last_upd > time_tol:

        # debug
        print("Updating the database...")

        # update db
        db = MtgDB(DB_NAME)
        db.scryfall_bulk_update()

        # update last update date
        year = str(today.year)
        if today.month < 10:
            mon = "0" + str(today.month)
        if today.day < 10:
            day = "0" + str(today.day)
        date_str = year + ' ' + mon + ' ' + day

        with open(upd_file, 'w') as f:
            f.write(date_str)

def get_db():
    '''
    Function that checks for update then fetches the scryfall database.
    '''

    # update the database
    update_db()
    db = MtgDB(DB_NAME)
    return db.root.scryfall_cards

def ord_encoding(df, column_list):
    '''
    Takes a df and a list of columns to encode, and returns the df
    with the requested columns encoded via ordinal encoding (and drops the originals.)

    Params:
    - df : pandas.DataFrame
        the dataframe to encode
    - colum_list : list of str
        list of columns to encode categorically
    '''
    for col in column_list:
        enc = OrdinalEncoder()
        df[col] = enc.fit_transform(np.array(df[col]).reshape(-1,1)).flatten()

    return df

def bin_targets(df, target_col='price'):
    '''
    Create classes for the target.

    Our target is price. The classes are as follows:
        0 : $0.00 <= p < $0.50
        1 : $0.50 <= p < $2.00
        2 : $2.00 <= p < $10.00
        3 : $10.00 <= p < $100.00
        4 : $100.00 <= p
    '''

    def target_price(row):
        # turn p into class
        p = float(row[target_col])

        if p < 0.5:
            return 0
        elif p < 2:
            return 1
        elif p < 10:
            return 2
        elif p < 100:
            return 3
        else:
            return 4
        
    col = 'class_' + target_col
    df[col] = df.apply(target_price, axis=1)
    return df

def get_df(save_csv=False, csv_name='mtg_data.csv'):
    '''
    Iterate through the database and pull data into pandas.DataFrame. Once a dataframe is formed, various
    engineering and cleanup is performed on the dataset.

    Params:
    - save_csvs : bool
        Save the finished dataframe as a csv file
    - csv_name : str
        Name of file to save datafram to if save_csv is true
    '''

    # fetch db
    db = get_db()
    N = len(db)
    loop = tqdm(total = N, position=0, leave=False)

    # init the list of attributes we want to fetch
    all_cards = list()
    
    for i in range(N):
        # get the card
        card = db[i]
        has_foil = False

        # update timer
        loop.update()

        # check for oversized, ignore as they are not 'normal' cards
        if card.oversized == True:
            continue

        # check for non-played cards
        if card.layout in LAYOUT_IGNORE:
            continue



        # check for foil finish
        finishes = getattr(card, 'finishes')
        prices = getattr(card, 'prices')

        if 'nonfoil' in finishes:
            card_data = []
            for att in ATTRIBUTES:
                card_data.append(getattr(card, att))

            # last two will be is_foil and price
            card_data.append(0)
            card_data.append(prices['usd'])

            # store the card
            all_cards.append(card_data)


        if 'foil' in finishes:
            card_data = []
            for att in ATTRIBUTES:
                card_data.append(getattr(card, att))

            # last two will be is_foil and price
            card_data.append(1)
            card_data.append(prices['usd_foil'])

            # store the card
            all_cards.append(card_data)


    # create the dataframe
    cols = ATTRIBUTES + ['is_foil', 'price']
    df = pd.DataFrame(all_cards, columns=cols)

    # log to console
    print("\nFinished Compiling DataFrame")
    print("Processing data...\n")

    # ---- feature engineering ----

    # get rid of things with no price
    df = df.dropna(axis=0, subset=['price'])

    cols_to_drop = []

    # binary colors
    has_list = ['has_white', 'has_blue', 'has_black', 'has_red', 'has_green']
    color_list = ['W', 'U', 'B', 'R', 'G']
    has_entry = lambda row, col_name, entry: 1 if entry in row[col_name] else 0

    for i, has in enumerate(has_list):
        df[has] = df.apply(has_entry, axis=1, args=('color_identity', color_list[i]))
    df['total_num_colors'] = np.sum(df[has_list], axis=1)
    cols_to_drop.append("color_identity")

    # length of oracle text
    def text_len(row):
        txt = row['oracle_text']
        if txt is None:
            return 0
        else:
            return len(txt)
    df['len_oracle_text'] = df.apply(text_len, axis=1)
    cols_to_drop.append("oracle_text")

    # produces mana
    def prod_mana(row):
        arr = ['produced_mana']
        if arr is None:
            return 0
        else:
            return len(arr)
    df['len_mana_types_produced'] = df.apply(prod_mana, axis=1)
    cols_to_drop.append("produced_mana")

    # has frame effect
    def has_frame(row):
        if row['frame_effects'] is None:
            return 0
        else:
            return 1
    df['has_frame_effect'] = df.apply(has_frame, axis=1)
    cols_to_drop.append('frame_effects')

    # has flavor text
    def has_flavor(row):
        if row['flavor_text'] is None:
            return 0
        else:
            return 1
    df['has_flavor_text'] = df.apply(has_flavor, axis=1)
    cols_to_drop.append("flavor_text")

    # binary types (artifact, enchantment, creature, etc)
    # TODO

    # binary keywords
    kw_to_keep = ['Flying', 'Enchant', 
                  'First strike', 'Equip', 
                  'Vigilance', 'Transform',
                  'Cycling', 'Haste', 
                  'Trample', 'Mill', 
                  'Flash', 'Scry']
    for i, kw in enumerate(kw_to_keep):
        cname = "has_kw_" + kw
        df[cname] = df.apply(has_entry, axis=1, args=('keywords', kw))

    # number of keywords
    num_kw = lambda row: len(row['keywords'])
    df["number_keywords"] = df.apply(num_kw, axis=1)
    cols_to_drop.append('keywords')

    # artist
    # TODO: there are over 1600, how do we deal with this?

    # encode rarity
    df = ord_encoding(df, ['rarity'])

    # fix nan values
    df['power_num'] = df['power_num'].fillna(-100) # -1 is already taken
    df['toughness_num'] = df['toughness_num'].fillna(-100)

    # set target classes
    df = bin_targets(df)

    # drop unnecessary columns
    df = df.drop(columns=cols_to_drop)

    if save_csv:
        print(f"Writing data to {csv_name}")
        df.to_csv(csv_name)

    print("Data Fetching COMPLETE")
    return df


if __name__ == "__main__":

    df = get_df(save_csv=True)
    print(df.head())

