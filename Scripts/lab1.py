import pandas as pd

df = pd.read_csv("e:\\Python\\lab1\\lab1\\train.csv");

print(df)

for col in df:
    print(col, '\t', df[col].isnull().sum())


from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

HomePlanet = df['HomePlanet'].mode();
df['HomePlanet'] = df['HomePlanet'].fillna(HomePlanet);

CryoSleep = df['CryoSleep'].median();
df['CryoSleep'] = df['CryoSleep'].fillna(CryoSleep);

Cabin = df['Cabin'].mode();
df['Cabin'] = df['Cabin'].fillna(Cabin.get(0));

Destination = df['Destination'].mode();
df['Destination'] = df['Destination'].fillna(Destination.get(0));

Age = df['Age'].mean();
df['Age'] = df['Age'].fillna(Age);
df['Age'] = scaler.fit_transform(df['Age'].to_numpy().reshape(-1,1))

VIP = df['VIP'].median();
df['VIP'] = df['VIP'].fillna(VIP);

RoomService = df['RoomService'].mean();
df['RoomService'] = df['RoomService'].fillna(RoomService);
df['RoomService'] = scaler.fit_transform(df['RoomService'].to_numpy().reshape(-1,1))

FoodCourt = df['FoodCourt'].mean();
df['FoodCourt'] = df['FoodCourt'].fillna(FoodCourt);
df['FoodCourt'] = scaler.fit_transform(df['FoodCourt'].to_numpy().reshape(-1,1))

ShoppingMall = df['ShoppingMall'].mean();
df['ShoppingMall'] = df['ShoppingMall'].fillna(ShoppingMall);
df['ShoppingMall'] = scaler.fit_transform(df['ShoppingMall'].to_numpy().reshape(-1,1))

Spa = df['Spa'].mean();
df['Spa'] = df['Spa'].fillna(Spa);
df['Spa'] = scaler.fit_transform(df['Spa'].to_numpy().reshape(-1,1))

VRDeck = df['VRDeck'].mean();
df['VRDeck'] = df['VRDeck'].fillna(VRDeck);
df['VRDeck'] = scaler.fit_transform(df['VRDeck'].to_numpy().reshape(-1,1))

Name = df['Name'].mode();
df['Name'] = df['Name'].fillna(Name.get(0));

for col in df:
    print(col, '\t', df[col].isnull().sum())


columns=['HomePlanet', 'Cabin', 'Destination', 'Name']
col_values_matrix = [[],[],[],[],[],[]];
for i in range(0,4):
    col = []
    for j in range(0, df[columns[i]].size) :
        val = df[columns[i]][j]
        if col_values_matrix[i].count(val) == 0 :
            col_values_matrix[i].append(val)
        
        col.append(col_values_matrix[i].index(val))
    df.insert(df.columns.size, columns[i]+"_int", col)
    df.pop(columns[i])
    
    
    
print(df)

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)


train_df.to_csv("preprocessed.csv", index=False)
test_df.to_csv("preprocessed_test.csv", index=False)