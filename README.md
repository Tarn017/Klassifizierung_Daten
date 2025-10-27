mkdir -p ~/tmp-pip
TMPDIR=~/tmp-pip pip install --no-cache-dir tensorflow 


# Nutzen eines Feedforward-Netzes (FFN):

  1. Importiert das Netz aus [NeuralNetwork](https://github.com/Tarn017/Klassifizierung_Daten/blob/main/NeuralNetwork.py): `from NeuralNetwork import FFN, validation_classification`
  
  3. **Funktion:** `FFN()`
     
     **Argumente:**  
     -	ordner = <Name des Ordners in dem die Daten liegen (falls sich Ordner und Skript sich am selben Speicherort befinden), sonst Pfad zum Ordner>  
     -	model_name = <Name für das Modell festlegen mit .keras Endung>  
     -	epochen = <Anzahl 'an Trainingsepochen>  
     -	n_full = < Form: [a,b,c,…], wobei a der Anzahl an Neuronen in der 1. Voll verbundenen Schicht entspricht, b der Anzahl in der 2. Schicht, usw. Es wird also die Anzahl der Schichten sowie deren Größe festgelegt>  
     -	resize = <Form: [Höhe,Breite], wobei Höhe und Breite Integer Werte sind und im Falle einer gewünschten Größenänderung der Bilder definiert werden können. Wird dieses Tupel nicht definiert, so werden die Bilder in Originalgröße verarbeitet>  
     -	val_ordner = <optinale Angabe falls ein Ordner mit Validierungsdaten vorhanden ist wird während des Trainings direkt die Performanz auf den Validierungsdaten angezeigt>  
     -	lr = <Lernrate des Netzes, meistens zwischen 0.1 und 0.0001>  
     -	decay = <True/False, soll die Lernrate während des Trainings abnehmen?>

     **Beispiel:**
     ```python
         FFN('fruits', 'fruit.keras',
            epochen=20,
            n_full=[200, 100, 50],
            resize=[128, 128],
            val_ordner='fruits_val',
            lr=0.001,
            decay=False
            )
      ```

  3. **Funktion:** validation_classification()

     **Argumente:**
     -	model = <Name des Modells mit .keras Endung das validiert werden soll>  
     -	val_ordner = <Name/Pfad des Ordners in dem die Bilder liegen mit denen das Modell getestet werden soll. Die Ordnerstruktur muss dabei derer des Ordners (ordner=…) entsprechen auf den das Modell trainiert wurde>  

     **Beispiel:**
     ```python
        acc = validation_classification(model='fruit.keras', val_ordner='fruits_val')
     ```

# Credits

Data from directory 'fruits': Ali Hasnain
