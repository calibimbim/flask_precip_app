from flask import Flask, render_template, request, send_file
import pandas as pd
import xgboost as xgb
import os
import io

app = Flask(__name__)

# === Load model and data ===
MODEL_PATH = os.path.join(os.getcwd(), "xgboost_model.json")
DATA_PATH = os.path.join(os.getcwd(), "df_predicts.csv")
LOC_PATH = os.path.join(os.getcwd(), "locations.csv")

xgb_model = xgb.XGBRegressor()
xgb_model.load_model(MODEL_PATH)

df_predict = pd.read_csv(DATA_PATH)
df_predict['datetime'] = pd.to_datetime(df_predict['datetime'])

locations_df = pd.read_csv(LOC_PATH)
location_names = sorted(locations_df['name'].unique())

# === GLOBAL to hold predictions as DataFrame ===
df_results = pd.DataFrame()

@app.route('/', methods=['GET', 'POST'])
def predict():
    global df_results
    predictions = []
    result_data = []

    if request.method == 'POST':
        selected_locations = request.form.getlist('locations')
        start_date_str = request.form.get('start_date')
        end_date_str = request.form.get('end_date')

        if not selected_locations or not start_date_str or not end_date_str:
            predictions.append(("Error", "Please select locations and both start and end dates."))
            return render_template('form.html', locations=location_names, predictions=predictions)

        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)

        if start_date > end_date:
            predictions.append(("Error", "Start date cannot be after end date."))
            return render_template('form.html', locations=location_names, predictions=predictions)

        features = [
            'tmean', 'trange', 'tmax', 'tmin',
            'rh', 'sh', 'wind', 'Latitude', 'Longitude'
        ]

        for location_name in selected_locations:
            loc_row = locations_df[locations_df['name'] == location_name]
            if loc_row.empty:
                predictions.append((location_name, "Invalid location."))
                continue

            lat = float(loc_row['Latitude'].values[0])
            lon = float(loc_row['Longitude'].values[0])

            subset = df_predict[
                (df_predict['Latitude'] == lat) &
                (df_predict['Longitude'] == lon) &
                (df_predict['datetime'] >= start_date) &
                (df_predict['datetime'] <= end_date)
            ]

            if subset.empty:
                predictions.append((location_name, "No data available for this range."))
                continue

            X = subset[features]
            y_preds = xgb_model.predict(X)

            for dt, pred in zip(subset['datetime'], y_preds):
                label = f"{location_name} - {dt.date()}"
                predictions.append((label, f"{pred:.2f} mm"))
                result_data.append({
                    'Location': location_name,
                    'Date': dt.date(),
                    'Predicted Rainfall (mm)': round(pred, 2)
                })

    # Store downloadable results
    df_results = pd.DataFrame(result_data)
    return render_template('form.html', locations=location_names, predictions=predictions)


@app.route('/download')
def download_csv():
    global df_results
    if df_results.empty:
        return "No prediction data available.", 400

    buffer = io.StringIO()
    df_results.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='rainfall_predictions.csv'
    )


if __name__ == '__main__':
    app.run(debug=True)
