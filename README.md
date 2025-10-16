# Nutzen eines Feedforward-Netzes (FFN):

  1. Importiert das Netz aus [NeuralNetwork](): `from NeuralNetwork import FFN, validation_classification`
  
  3. **Funktion:** `FFN()`
     
     **Argumente:**  
     -	ordner = <Name des Ordners in dem die Daten liegen (falls sich Ordner und Skript sich am selben Speicherort befinden), sonst Pfad zum Ordner>  
     -	model_name = <Name für das Modell festlegen mit .keras Endung>  
     -	epochen = <Anzahl 'an Trainingsepochen>  
     -	n_full = < Form: [a,b,c,…], wobei a der Anzahl an Neuronen in der 1. Voll verbundenen Schicht entspricht, b der Anzahl in der 2. Schicht, usw. Es wird also die Anzahl der Schichten sowie deren Größe festgelegt>  
     -	droprate = <Wert zwischen 0 und 1 für die Dropoutwahrscheinlichkeit>  
     -	resize = <Form: [Höhe,Breite], wobei Höhe und Breite Integer Werte sind und im Falle einer gewünschten Größenänderung der Bilder definiert werden können. Wird dieses Tupel nicht definiert, so werden die Bilder in Originalgröße verarbeitet>  
     -	val_ordner = <optinale Angabe falls ein Ordner mit Validierungsdaten vorhanden ist wird während des Trainings direkt die Performanz auf den Validierungsdaten angezeigt>  
     -	aug_parameter = <Optionale Angabe der Form [spiegeln,rotation,zoom,kontrast], falls Data Augmentation angewendet werden soll. Fur das Spiegeln sind die Werte 'vertical', 'horizontal' und 'horizontal_and_vertical' zugelassen. Die anderen drei Werte                        werden als Zahl zwischen 0 und 1 gewählt. ['horizontal', 0.1, 0.2, 0.3] entspricht der Einstellung dass manche Bilder horizontal gespiegelt werden, manche um 0.1*360=36 Grad gedreht werden, in anderen wird um 20% gezoomt und manchmal wird der kontrast                    zufällig aus [1−0.3,1+0.3]=[0.7,1.3] gewählt.>  
     -	alpha = <optional, falls weight decay Regularisierung genutzt werden soll. Alpha entspricht dem Gewicht des Regularisierungsterms>  
     -	lr = <Lernrate des Netzes, meistens zwischen 0.1 und 0.0001>  
     -	decay = <True/False, soll die Lernrate während des Trainings abnehmen?>

     **Beispiel:**
     ```python
         FFN('fruits', 'fruit.keras',
            epochen=20,
            n_full=[200, 100, 50],
            droprate=0.4,
            resize=[128, 128],
            val_ordner='fruits_val',
            aug_parameter=['vertical', 0.05, 0.05, 0.1],
            alpha=0.0002,
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
