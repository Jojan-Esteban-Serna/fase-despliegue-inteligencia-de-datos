import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import VotingClassifier
import joblib

list_products = joblib.load('list_products.pkl')
kmeans = joblib.load('kmeans.pkl')
votingC = joblib.load('votingC.pkl')

def obtener_recomendaciones(description, unit_price, quantity, basket_price):
    df_cleaned = pd.DataFrame({'Description': [description], 'UnitPrice': [unit_price], 'Quantity': [quantity], 'BasketPrice': [basket_price]})
    liste_produits = df_cleaned['Description']
    X = pd.DataFrame()
    for key, occurence in list_products:
        X.loc[:, key] = list(map(lambda x: int(key.upper() in x), liste_produits))

    threshold = [0, 1, 2, 3, 5, 10]
    label_col = []
    for i in range(len(threshold)):
        if i == len(threshold) - 1:
            col = '.>{}'.format(threshold[i])
        else:
            col = '{}<.<{}'.format(threshold[i], threshold[i + 1])
        label_col.append(col)
        X.loc[:, col] = 0

    for i, prod in enumerate(liste_produits):
        prix = df_cleaned[df_cleaned['Description'] == prod]['UnitPrice'].mean()
        j = 0
        while prix > threshold[j]:
            j += 1
            if j == len(threshold): break
        X.loc[i, label_col[j - 1]] = 1

    n_cluster = kmeans.predict(X)[0]
    for i in range(5):
        df_cleaned.loc[:, 'categ_{}'.format(i)] = 100 if i == n_cluster else 0

    st.write("Se corrio el modelo de clustering y asi se ven los datos con los respectivos ajustes")
    st.write(df_cleaned.head())

    new_dataframe = pd.DataFrame({
        'mean': [unit_price * quantity],
        **{f'categ_{i}': df_cleaned[f'categ_{i}'] for i in range(5)}
    })

    st.write("Se modificaron los datos para que se puedan predecir las categorias de los clientes y se ven asi")
    st.write(new_dataframe.head())

    cluster = votingC.predict(new_dataframe)[0]
    return cluster


if __name__ == '__main__':
    st.title('Prediccion de categoria de cliente')
    st.header('Ingresa los datos del producto')
    description = st.text_input("Descripcion del producto")
    unit_price = st.number_input("Precio Unitario", min_value=0.0, step=0.5)
    quantity = st.number_input("Cantidad", min_value=0, step=1)

    if st.button('Generar recomendación'):
        basket_price = unit_price * quantity
        st.write(f"El precio de la cesta es de {basket_price:.2f}")

        # Obtener recomendaciones
        cluster = obtener_recomendaciones(description, unit_price, quantity, basket_price)

        # Mostrar recomendaciones en una tabla
        st.write("Se ejecuto el modelo de clasificacion y fue asignado al cluster {}".format(cluster))
