# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 14:23:10 2026

@author: Lenov X13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_preprocess_data(file_path):
    """Charge et prétraite les données du fichier CSV D_clean."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    df['date_entree'] = pd.to_datetime(df['date_entree'], errors='coerce')
    df['Heure_entree'] = df['date_entree'].dt.hour
    df['Jour_semaine'] = df['date_entree'].dt.day_name()
    df['Date'] = df['date_entree'].dt.date

    # Tranches d'âge dérivées de la colonne 'age' (numérique continue)
    bins = [0, 5, 15, 25, 40, 60, 75, np.inf]
    labels = ['0-4', '5-14', '15-24', '25-39', '40-59', '60-74', '75+']
    df['tranche_age'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    return df


def analyze_patient_flow(df):
    print("\n" + "=" * 50)
    print("Analyse du Flux des Patients")
    print("=" * 50)

    total_patients = df['dossier'].nunique()
    print(f"\nNombre total de patients (dossiers uniques): {total_patients}")

    print("\nRépartition par année:")
    print(df['annee'].value_counts().sort_index())

    print("\nRépartition par sexe:")
    print(df['sexe'].value_counts(dropna=False))

    print("\nRépartition par tranche d'âge:")
    print(df['tranche_age'].value_counts().sort_index())

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Heure_entree', data=df, hue='annee', palette='viridis')
    plt.title("Distribution des Admissions par Heure")
    plt.xlabel("Heure de la Journée")
    plt.ylabel("Nombre de Patients")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/admissions_par_heure.png", dpi=120)
    plt.close()

    daily_counts = df.groupby('Date').size()
    plt.figure(figsize=(12, 6))
    daily_counts.plot(kind='line', marker='o')
    plt.title("Nombre de Patients par Jour")
    plt.xlabel("Date")
    plt.ylabel("Nombre de Patients")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/admissions_par_jour.png", dpi=120)
    plt.close()


def analyze_diagnostics(df):
    print("\n" + "=" * 50)
    print("Analyse des Diagnostics")
    print("=" * 50)

    if 'diagnostic_categorie' not in df.columns:
        print("Aucune colonne de diagnostic trouvée.")
        return

    top_diagnostics = df['diagnostic_categorie'].value_counts().head(10)
    print("\nTop 10 des catégories diagnostiques:")
    print(top_diagnostics)

    plt.figure(figsize=(12, 8))
    top_diagnostics.sort_values().plot(kind='barh', color='teal')
    plt.title("Top 10 des Catégories Diagnostiques")
    plt.xlabel("Nombre de Cas")
    plt.ylabel("Diagnostic")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/top_diagnostics.png", dpi=120)
    plt.close()


def analyze_trauma_cases(df):
    print("\n" + "=" * 50)
    print("Analyse des Cas de Trauma")
    print("=" * 50)

    trauma_cases = df[df['categorie_diagnostique_large'] == 'Trauma']
    print(f"\nNombre de cas de trauma identifiés: {len(trauma_cases)}")

    if not trauma_cases.empty and 'trauma_type' in df.columns:
        causes = trauma_cases['trauma_type'].value_counts()
        print("\nCauses des trauma:")
        print(causes)

        plt.figure(figsize=(10, 6))
        causes.plot(
            kind='pie', autopct='%1.1f%%',
            colors=sns.color_palette('pastel', len(causes))
        )
        plt.title("Répartition des Causes de Trauma")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/causes_trauma.png", dpi=120)
        plt.close()


def analyze_services(df):
    print("\n" + "=" * 50)
    print("Analyse par Services")
    print("=" * 50)

    service_dist = df['service_principal'].value_counts()
    print("\nRépartition par service:")
    print(service_dist)

    plt.figure(figsize=(12, 8))
    service_dist.plot(kind='bar', color='purple')
    plt.title("Répartition des Patients par Service")
    plt.xlabel("Service")
    plt.ylabel("Nombre de Patients")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/repartition_services.png", dpi=120)
    plt.close()


def analyze_outcomes(df):
    print("\n" + "=" * 50)
    print("Analyse des Évolutions Cliniques et Séjours")
    print("=" * 50)

    print("\nTypologie des cas:")
    print(df['typologie_cas'].value_counts(dropna=False))

    print("\nÉvolution immédiate:")
    print(df['evolution_immediate'].value_counts(dropna=False))

    print("\nDurée d'hospitalisation:")
    print(df['duree_hospitalisation'].value_counts(dropna=False))

    plt.figure(figsize=(8, 6))
    df['evolution_immediate'].value_counts(dropna=False).plot(kind='bar', color='orange')
    plt.title("Évolution Immédiate des Patients")
    plt.xlabel("Évolution")
    plt.ylabel("Nombre de Patients")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/evolution_immediate.png", dpi=120)
    plt.close()


def generate_full_report(df):
    analyze_patient_flow(df)
    analyze_diagnostics(df)
    analyze_trauma_cases(df)
    analyze_services(df)
    analyze_outcomes(df)


def main():
    file_path = "D_clean.csv"
    try:
        df = load_and_preprocess_data(file_path)
        generate_full_report(df)
        df.to_csv('magal_2024_analyzed.csv', index=False)
        print("\nAnalyse terminée. Données sauvegardées dans 'magal_2024_analyzed.csv'")
        print(f"Graphiques sauvegardés dans le dossier '{OUTPUT_DIR}/'")
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")


if __name__ == "__main__":
    main()
    
    
    

#==================================================
#Analyse du Flux des Patients
#==================================================

#Nombre total de patients (dossiers uniques): 1904

#Répartition par année:
#annee
#2023     644
#2024    1260
#Name: count, dtype: int64

#Répartition par sexe:
#sexe
#Femme    932
#Homme    749
#NaN      223
#Name: count, dtype: int64
#
#Répartition par tranche d'âge:
#tranche_age
#0-4       82
#5-14     197
#15-24    241
#25-39    350
#40-59    321
#60-74    120
#75+       25
#Name: count, dtype: int64

#==================================================
#Analyse des Diagnostics
#==================================================

#Top 10 des catégories diagnostiques:
#diagnostic_categorie
#DIGESTIF                 198
#TRAUMA                   181
#OPHTALMOLOGIQUE          113
#BUCCODENTAIRE             92
#MATERNITE_GYNECOLOGIE     90
#NEUROPSYCHIATRIQUE        90
#DERMATOLOGIQUE            89
#ORL                       81
#RESPIRATOIRE              80
#CARDIOVASCULAIRE          55
#Name: count, dtype: int64

#==================================================
#Analyse des Cas de Trauma
#==================================================

#Nombre de cas de trauma identifiés: 180

#Causes des trauma:
#trauma_type
#Accident de la circulation    74
#Accident domestique           72
#Accident moto                 13
#Coups et blessures            12
#Accident charrette             9
#Name: count, dtype: int64

#==================================================
#Analyse par Services
#==================================================

#Répartition par service:
#service_principal
#Consultation Externe                     836
#Service dAccueil et dUrgence             349
#Pédiatrie                                202
#Ophtalmologie                            108
#Odonto-stomatologie                       86
#Gynécologie-Obstétrique                   68
#Oto-Rhino-Laryngologie (ORL)              48
#Neurochirurgie                            39
#Dermatologie                              29
#Urologie-Andrologie                       25
#Neurologie                                13
#Urgences gynéco-obstétricales             10
#Chirurgie pediatrique                      9
#Médecine interne                           6
#Endocrinologie-Diabétologie-Nutrition      4
#Réanimation                                4
#Cardiologie                                3
#Autres                                     2
#Orthopédie                                 1
#Name: count, dtype: int64

#==================================================
#Analyse des Évolutions Cliniques et Séjours
#==================================================

#Typologie des cas:
#typologie_cas
#cas simple    1518
#NaN            283
#cas grave      103
#Name: count, dtype: int64

#Évolution immédiate:
#evolution_immediate
#NaN                1483
#favorable           339
#non appréciable      74
#défavorable           4
#décès                 4
#Name: count, dtype: int64

#Durée d'hospitalisation:
#duree_hospitalisation
#NaN     1672
#0-3j     175
#4-7j      39
#+7j       18
#Name: count, dtype: int64

#Analyse terminée. Données sauvegardées dans 'magal_2023_2024_analyzed.csv'
#Graphiques sauvegardés dans le dossier 'figures/'
