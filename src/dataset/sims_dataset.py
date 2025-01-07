import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split



def create_sims_dataset(path = 'C:\\Users\\lollo\\Lavoro\\Ricerca\\energiai\\dataset\\sims\\TSA.csv',
                         train_size=0.6, val_size=0.2, test_size=0.2, timestep = 300, offset =100):

    white_list = [
        # 'x',
        # 'y',
        'bedroom',
        'livingroom',
        'kitchen',
        'bathroom',
        'elapsed_time_cumulative',
        'elapsed_time',
        'exceeds_average'  
    ]

    df = pd.read_csv(path)
    df['action_group'] = (df['room'] != df['room'].shift()).cumsum()
    action_group_elapsed_time = df.groupby('action_group').agg(
        start_elapsed=('elapsed_time', 'first'),
        end_elapsed=('elapsed_time', 'last'),
        room=('room', 'first')
    )

    action_group_elapsed_time['elapsed_time_diff'] = (
        action_group_elapsed_time['end_elapsed'] - action_group_elapsed_time['start_elapsed']
    )

    action_group_elapsed_time['elapsed_time_hours'] = action_group_elapsed_time['elapsed_time_diff'] / 3600

    elapsed_time_stats = action_group_elapsed_time.groupby('room')['elapsed_time_hours'].agg(
        average='mean',
        maximum='max',
        minimum='min'
    ).reset_index()

    average_elapsed_time_by_room = elapsed_time_stats.set_index('room')['average'].to_dict()

    df['elapsed_time_cumulative'] = df.groupby('action_group')['elapsed_time'].transform(lambda x: x - x.iloc[0])

    df['exceeds_average'] = df.apply(
        lambda row: 1 if row['elapsed_time_cumulative'] / 3600 > average_elapsed_time_by_room.get(row['room'], 0) else 0,
        axis=1
    )
    interval = 1  # seconds
    df['exist'] = [i % interval == 0 for i in range(len(df))]
    df = df[df['exist']]

    # df['room_change'] = df['room'] != df['room'].shift(-1)

    # df = df[df['room_change']].drop(columns=['room_change'])
    df = df[white_list]
    # print(df)


    # return df

    # print(df)

    train_len = int(len(df) * train_size)
    val_len = int(len(df) * val_size)

    train_set = df.iloc[:train_len]
    # print('train:', train_set)

    val_set = df.iloc[train_len:train_len + val_len]
    # print('val:',val_set)

    test_set = df.iloc[train_len + val_len:]
    # print('test:',test_set)


    train = SimsDataset(train_set, timestep = timestep, offset = offset)
    val = SimsDataset(val_set, timestep = timestep, offset = offset)
    test = SimsDataset(test_set, timestep = timestep, offset = offset)


    return train, val, test



class SimsDataset(Dataset):
    def __init__(self, dataframe, timestep = 300, offset = 100):
        self.timestep = timestep
        self.offset = offset
        self.dataframe = dataframe
        self.n_features = dataframe.shape[-1] - 1 
       
    def __len__(self):
        return int((len(self.dataframe) - self.timestep) / self.offset)

    def __getitem__(self, idx):
        start = idx * self.offset
        end = start + self.timestep
        data = self.dataframe.iloc[start:end, :].values.astype('float32')
        # x = data[:, :-1]
        # y = data[:-1, -1:]
        x_curr = data[:-1, :-1]
        x_next = data[1:, :-3]
        # print('x_next è pari a:', x_next, '\n\nx_curr è pari a:', x_curr)
        y = data[1:, -1:]
        return torch.tensor(x_curr),torch.tensor(x_next),  torch.tensor(y)
    # , torch.tensor(label)
    def num_feat(self):
        return self.n_features

    def out_feat(self):
        return 4

    if __name__ == '__main__':
        df = create_sims_dataset()
        print(df)