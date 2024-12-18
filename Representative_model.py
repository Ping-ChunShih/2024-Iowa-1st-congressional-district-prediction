import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from census import Census
from us import states

# 初始化 Census API
API_KEY = 'd7b114eba874afb42cf77d6c9933460b6e292b12'
c = Census(API_KEY)

# 測試資料集獲取函數
def get_demographics_data(state, district, year):
    state_fips = states.lookup(state).fips

    data = c.acs5.state_congressional_district(
        ('NAME', 'B01003_001E',
        'B02001_002E',
        'B02001_003E',
        'B02001_004E',
        'B02001_005E',
        'B02001_006E',
        'B02001_007E',
        'B02001_008E',
        'B01001_007E',
        'B01001_008E',
        'B01001_009E',
        'B01001_010E',
        'B01001_011E',
        'B01001_012E',
        'B01001_013E',
        'B01001_014E', 
        'B01001_015E',
        'B01001_016E',
        'B01001_017E',
        'B01001_018E',
        'B01001_019E',
        'B01001_020E',
        'B01001_021E',
        'B01001_022E',
        'B01001_023E',
        'B01001_024E',
        'B01001_025E',
        'B01001_002E',
        'B01001_026E',
        'B19013_001E',
        'B15003_001E',
        'B23025_005E'),
        state_fips, district, year=year)

    df = pd.DataFrame(data)
    df = df.rename(columns={
        'NAME': 'District',
        'B01003_001E': 'Total_Population',
        'B02001_002E': 'White_Alone',
        'B02001_003E': 'Black_or_African_American_Alone',
        'B02001_004E': 'American_Indian_and_Alaska_Native_Alone',
        'B02001_005E': 'Asian_Alone',
        'B02001_006E': 'Native_Hawaiian_and_Other_Pacific_Islander_Alone',
        'B02001_007E': 'Some_Other_Race_Alone',
        'B02001_008E': 'Two_or_More_Races',
        'B01001_002E': 'Male_Population',
        'B01001_026E': 'Female_Population',
        'B19013_001E': 'Median_Household_Income',
        'B15003_001E': 'Educational_Attainment',
        'B23025_005E': 'Unemployment'
    })

    df['18-34'] = (df['B01001_007E'] + df['B01001_008E'] + df['B01001_009E'] +
                   df['B01001_010E'] + df['B01001_011E'] + df['B01001_012E'])

    df['35-64'] = (df['B01001_013E'] + df['B01001_014E'] + df['B01001_015E'] +
                   df['B01001_016E'] + df['B01001_017E'] + df['B01001_018E'] +
                   df['B01001_019E'])

    df['65 and older'] = (df['B01001_020E'] + df['B01001_021E'] + df['B01001_022E'] +
                          df['B01001_023E'] + df['B01001_024E'] + df['B01001_025E'])

    df['Male_Percentage'] = (df['Male_Population'] / df['Total_Population']) * 100
    df['Female_Percentage'] = (df['Female_Population'] / df['Total_Population']) * 100

    df['White_Percentage'] = (df['White_Alone'] / df['Total_Population']) * 100
    df['Black_Percentage'] = (df['Black_or_African_American_Alone'] / df['Total_Population']) * 100
    df['American_Indian_Percentage'] = (df['American_Indian_and_Alaska_Native_Alone'] / df['Total_Population']) * 100
    df['Asian_Percentage'] = (df['Asian_Alone'] / df['Total_Population']) * 100
    df['Native_Hawaiian_Percentage'] = (df['Native_Hawaiian_and_Other_Pacific_Islander_Alone'] / df['Total_Population']) * 100
    df['Other_Race_Percentage'] = (df['Some_Other_Race_Alone'] / df['Total_Population']) * 100
    df['Two_or_More_Races_Percentage'] = (df['Two_or_More_Races'] / df['Total_Population']) * 100

    df = df[['Total_Population', '18-34', '35-64', '65 and older',
             'Male_Population', 'Female_Population', 'Male_Percentage',
             'Female_Percentage', 'White_Percentage', 'Black_Percentage',
             'American_Indian_Percentage', 'Asian_Percentage',
             'Native_Hawaiian_Percentage', 'Other_Race_Percentage',
             'Two_or_More_Races_Percentage', 'Median_Household_Income',
             'Educational_Attainment', 'Unemployment']]
    return df


file_path = 'training data.csv'
training_data = pd.read_csv(file_path)

label_encoder = LabelEncoder()
training_data['Representative_Winner'] = label_encoder.fit_transform(training_data['Representative_Winner'])

X = training_data.drop(columns=['Year', 'District', 'Representative_Winner'])
y = training_data['Representative_Winner']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_encoded = to_categorical(y)


model = Sequential([
    Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y_encoded, epochs=100, batch_size=32)

test_data = get_demographics_data('IA', '01', 2022)
test_data_scaled = scaler.transform(test_data)

predictions = model.predict(test_data_scaled)
predicted_classes = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

test_data['Predicted_Winner'] = predicted_classes
print(test_data[['Predicted_Winner']])
