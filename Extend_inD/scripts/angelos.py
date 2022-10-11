import pandas as pd 
import os
import numpy as np
import pathlib
import evaluation
import os



# Regarding the formatting in results/AMENet/predictions_first/*.txt:
# columns = [frame,originalObjectId,xCenter,yCenter]
# For every originalObjectId there is 20 rows, where the first 8 rows are 
# inputs to model while the last 12 rows are the 1-step prediction,
# 2-step prediction, ..., 12-step prediction.

def load_data(path, recordingIds):
    df = []
    largest_objectId = 0
    n_objects_total = 0
    for filename in np.sort(os.listdir(path)): # To make more robust you could load according to the order in recordingId, but this requires parsing of filename
        if filename[-3:]=='npy':
            continue
        recId = int(filename.split('_')[-2])
        if ~np.isin(recId,recordingIds):
            continue
        tmp = pd.read_csv(path  / filename,delim_whitespace=True
                    ,names=['frame','originalObjectId','xCenter','yCenter'])
        tmp['recordingId'] = recId
        originalObjectIds = tmp['originalObjectId'].unique()
        tmp['locationId'] = get_locationId(recId)
        tmp = tmp.sort_values(['recordingId','originalObjectId','frame'],axis=0) ########
        tmp['objectId'] = np.nan
        nextObjectId = largest_objectId + 1
        n_objects = len(originalObjectIds)
        objectIds = range(nextObjectId,nextObjectId+n_objects)
        for originalId, id in zip(originalObjectIds,objectIds):
            tmp.loc[tmp['originalObjectId'] ==originalId,'objectId'] = id
        largest_objectId = largest_objectId +n_objects
        n_objects_total += n_objects
        df.append(tmp)
    df = pd.concat(df)
    assert(np.all(df.groupby('objectId').count()==20))
    assert(n_objects_total==df['objectId'].iloc[-1])
    return df


def get_locationId(recordingId):
    if recordingId in range(7,18):
        return 1
    elif recordingId in range(18,30):
        return 2
    elif recordingId in range(30,33):
        return 3
    elif recordingId in range(7):
        return 4

def sanity_ADE_FDE(df):
    ADEs = {}
    FDEs = {}
    for locId in range(1,5):
        ADEs[locId] = []
        FDEs[locId] = []
        tmp = df[df['locationId']==locId]
        for objId in tmp['objectId'].unique():
            idx = tmp['objectId'] == objId
            ADEs[locId].append(tmp.loc[idx,'errors'].mean())
            FDEs[locId].append(tmp.loc[idx,'errors'].iloc[-1])
        ADEs[locId] = np.mean(ADEs[locId])
        FDEs[locId] = np.mean(FDEs[locId])
    print(ADEs)
    print(FDEs)

def their_code(dfs):
    ADEs = {}
    FDEs = {}
    for (locId,df),(_,df_pred) in zip(dfs['truth'].groupby('locationId'),dfs['predictions'].groupby('locationId')):
        n_objects = len(df['objectId'].unique())
        truth = np.full((n_objects,12,4),np.nan)# n_traj,12,4
        for i,(_,object) in enumerate(df.groupby('objectId')):
            truth[i,:,2:] = object[['xCenter','yCenter']].to_numpy()
        pred = np.full((n_objects,1,12,2),np.nan)
        for i,(_,object) in enumerate(df_pred.groupby('objectId')):
            pred[i,0,:,:] = object[['xCenter','yCenter']].to_numpy()
        #print(locId)
        #errors = evaluation.get_errors(truth, pred)
        errors  = evaluation.get_evaluation(truth, pred, pred.shape[1])
        ADEs[locId] = errors[0, 2]
        FDEs[locId] = errors[1, 2]
    print(f'{ADEs}')
    print(FDEs)

def main(model):
    recordingIds = [5,6,14,15,16,17,26,27,28,29,32]
    paths = {'predictions': pathlib.Path(__file__).parent / f'../results/{model}'
            ,'truth':pathlib.Path(__file__).parent / '../trajectories_InD'}
    dfs = {}
    for case,path in paths.items():
        tmp = load_data(path,recordingIds)
        dfs[case] = tmp
        dfs[case] = tmp.groupby('objectId').apply(lambda g: g.iloc[8:])
        dfs[case] = dfs[case].droplevel('objectId')  

    df = dfs['predictions']
    df['errors'] = np.linalg.norm(dfs['predictions'][['xCenter','yCenter']].to_numpy()-dfs['truth'][['xCenter','yCenter']].to_numpy(),axis=1)
    ADEs = df.groupby(['locationId','objectId'])\
            .apply(lambda g: np.average(g['errors']))\
            .mean(level='locationId')
    print(ADEs)
    FDEs = df.groupby(['locationId','objectId'])\
            .apply(lambda g: g['errors'].iloc[-1])\
                .mean(level='locationId')
    print(FDEs)

    their_code(dfs)



if __name__ == '__main__':
    #main('AMENet')
    main('DCENet')