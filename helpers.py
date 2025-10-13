import pandas as pd
def save_to_log(data, message):
    with open("/home/gaen/Documents/codespace-gaen/Ts-master/experimental_models/log.txt", 'a') as f:
        f.write(message + '\n')
        f.write(str(data) + '\n\n\n')
def save_df(data, message):
    with open("/home/gaen/Documents/codespace-gaen/Ts-master/experimental_models/log.txt", 'a') as f:
        f.write(message + '\n')
        # Convert to DataFrame if data is not already a DataFrame
        if isinstance(data, pd.DataFrame):
            data.to_csv(f, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_csv(f, index=False)
        f.write('\n\n\n')