#!/usr/bin/env python3

import os

estensioni_desiderate = {'.dat', '.mdat', '.py', '.ana', '.plot', '.csv', '.pkl', '.h5',
                         '.svg', '.mdb', '.rul', '.visu', '.cc', '.cfg'}

try:
    # Ottieni il percorso corrente di lavoro
    cartella_di_lavoro = os.getcwd()

    # Itera sui file nella cartella di lavoro
    for filename in os.listdir(cartella_di_lavoro):
        percorso_file = os.path.join(cartella_di_lavoro, filename)

        # Verifica se il percorso è un file e se l'estensione non è desiderata
        if os.path.isfile(percorso_file) and os.path.splitext(filename)[1] not in estensioni_desiderate:
            os.remove(percorso_file)
            print(f"File eliminato: {filename}")

    print("Eliminazione completata.")
except Exception as e:
    print(f"Si è verificato un errore durante l'eliminazione: {str(e)}")
