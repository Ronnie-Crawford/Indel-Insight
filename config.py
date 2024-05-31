import json

class Config:
    
    def __init__(self, path: str = "./config.json"):
        
        self.set_config_from_file(path)
        
    def set_config_from_file(self, path: str) -> None:
        
        """
        Based on supplied path, reads and changes variables as necessary.
        """

        try:
            
            with open(path, "r") as config_file:
                
                print("Reading config file...")
                
                settings = json.load(config_file)
                for key, value in settings.items():
                    
                    setattr(self, key, value)

                self.path = config_file.name

        except FileNotFoundError: print("ERROR Config not found.")
        except Exception as e: print("ERROR.")
        
    def view_config_settings(self):
        
        """
        Prints current location of config file and values within.
        """
        
        try:
            
            with open(self.path, "r") as config_file:
                
                print(self.path)
                
                settings = json.loads(config_file)
                
                print("\n--- Config ---")
                for key, value in settings.items() : print(f"{key} : {value}")
                print("--------------\n")

        except FileNotFoundError : print("ERROR Config not found.")
        except Exception as e : print("ERROR.")
        
    def change_config_setting(self, key, value):
        
        """
        Assigns given value to given key within config file.
        """
        
        try:
            
            with open(self.path, "r") as config_file:
                
                settings = json.loads(config_file)

        except FileNotFoundError: print("ERROR Config not found.")
        except Exception as e: print("ERROR.")
        
        settings[key] = value
        
        try:
            with open(self.path, "w") as config_file:
                json.dumps(settings)
                
        except FileNotFoundError: print("ERROR Config not found.")
        except Exception as e: print("ERROR.")