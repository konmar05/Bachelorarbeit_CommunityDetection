class Vereine:

    def __setitem__(self, key, value):
        self.vereine[key] = value

    def vereine_hinzufuegen(self, daten):

        aktueller_verein = 1
        vereinsname = 2
        von = 3
        bis = 4

        for i in range(0, len(daten)):

            if (len(daten) <= 2):
                self.vereine[2023] = daten[aktueller_verein]
                break

            if (daten[von] == daten[bis]):
                self.vereine[daten[von]] = daten[vereinsname]
            else:
                for k in range(daten[von], daten[bis]):
                    self.vereine[k] = daten[vereinsname]

            vereinsname = vereinsname+3
            von = von+3
            bis = bis+3

            if (bis >= len(daten)):
                for index in range(daten[von], 2023):
                    self.vereine[index] = daten[vereinsname]
                self.vereine[2023] = daten[aktueller_verein]
                break

    def __init__(self, liste):
        self.vereine = {}
        self.vereine_hinzufuegen(liste)

    def __str__(self):
        return str(self.vereine)
