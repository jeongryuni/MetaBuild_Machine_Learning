import pandas as pd

df = pd.DataFrame({
    '날짜': ['2025-10-31', '2023-07-01', '2023-07-02']
})

df['날짜'] = pd.to_datetime(df['날짜'])

print(f"year       : {df['날짜'].dt.year.tolist()}")
print(f"month      : {df['날짜'].dt.month.tolist()}")
print(f"day        : {df['날짜'].dt.day.tolist()}")
print(f"dayofweek  : {df['날짜'].dt.dayofweek.tolist()}")   # 0=Monday, 6=Sunday
print(f"month_name : {df['날짜'].dt.month_name().tolist()}")
print(f"day_name   : {df['날짜'].dt.day_name().tolist()}")
