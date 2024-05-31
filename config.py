import json

class Config:
    
    def __init__(self, path: str = "./config.json"):

        """
        Initializes the Config object and sets configuration from the specified file.

        Parameters:
        path (str): The path to the configuration file. Defaults to './config.json'.
        """
        
        self.set_config_from_file(path)
        
    def set_config_from_file(self, path: str) -> None:
        
        """
        Reads the configuration file from the specified path and sets object attributes based on the file content.

        Parameters:
        path (str): The path to the configuration file.
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
        Prints the current location of the configuration file and its contents.
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
        Changes the value of a specific configuration setting in the file.

        Parameters:
        key (str): The configuration key to change.
        value: The new value for the configuration key.
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
