from app import create_app
from app.database import create_doctor_table
from app.database import create_doctor_table, create_patient_table
from app.database import add_image_path_column
from app.database import add_results_column

app = create_app()
create_doctor_table()    # 👈 This will create the "doctors" table when you run the app
create_patient_table()
#add_image_path_column()
#add_results_column()
if __name__ == '__main__':
    app.run(debug=True)
