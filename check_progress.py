import http.client
import json
import time
import requests

# Refer to https://docs.bfl.ml/ for details on finetuning progress parameters.

class FineTuneMonitor:
    def __init__(self, api_key: str, host: str = "api.us1.bfl.ai"):
        self.api_key = api_key
        self.host = host
   
    def check_progress(self, finetune_id: str) -> dict:
        """Check progress of a fine-tuning job."""
        conn = http.client.HTTPSConnection(self.host)
        headers = {
            'Content-Type': 'application/json',
            'X-Key': self.api_key
        }
        
        try:
            # First get the details to check if model exists
            endpoint = f"/v1/finetune_details?finetune_id={finetune_id}"
            print(f"\nChecking details at: {self.host}{endpoint}")
            
            conn.request("GET", endpoint, headers=headers)
            res = conn.getresponse()
            data = res.read()
            
            try:
                details_result = json.loads(data.decode('utf-8'))
                print("Details Response:")
                print(json.dumps(details_result, indent=2))
                
                if 'finetune_details' not in details_result:
                    return {
                        'status': 'Not Found',
                        'error': 'Model does not exist'
                    }
                    
            except json.JSONDecodeError:
                print("Raw Response:")
                print(data.decode('utf-8'))
                return None

            # Then check status for in-progress information
            endpoint = f"/v1/get_result?id={finetune_id}"
            print(f"\nChecking status at: {self.host}{endpoint}")
            
            conn = http.client.HTTPSConnection(self.host)
            conn.request("GET", endpoint, headers=headers)
            res = conn.getresponse()
            data = res.read()
            
            try:
                status_result = json.loads(data.decode('utf-8'))
                print("Status Response:")
                print(json.dumps(status_result, indent=2))
                
                # If task not found but we have details, it means training is complete
                if status_result.get('status') == 'Task not found':
                    status_result = {
                        'status': 'Ready',  # Model exists and training is done
                        'progress': '100%',
                        'result': 'Training completed'
                    }
                    
            except json.JSONDecodeError:
                print("Raw Response:")
                print(data.decode('utf-8'))
                status_result = {
                    'status': 'Ready',  # Default to ready if we have details
                    'progress': '100%'
                }

            # Combine the results
            combined_result = {
                'status': status_result.get('status', 'Ready'),
                'progress': status_result.get('progress'),
                'error': status_result.get('error'),
                'details': details_result.get('finetune_details', {}),
                'is_completed': status_result.get('status') in ['Ready', 'Task not found']
            }
            
            return combined_result
            
        except Exception as e:
            print(f"Error in check_progress: {str(e)}")
            return None
        finally:
            conn.close()
            
    def list_finetunes(self) -> dict:
        """List all fine-tunes."""
        conn = http.client.HTTPSConnection(self.host)
        headers = {
            'Content-Type': 'application/json',
            'X-Key': self.api_key
        }
        
        try:
            endpoint = "/v1/my_finetunes"
            print(f"\nListing fine-tunes from: {self.host}{endpoint}")
            
            conn.request("GET", endpoint, headers=headers)
            res = conn.getresponse()
            data = res.read()
            
            return json.loads(data.decode('utf-8'))
        finally:
            conn.close()

def monitor_progress(api_key: str, finetune_id: str, check_interval: int = 30):
    monitor = FineTuneMonitor(api_key)
    
    print(f"Starting to monitor fine-tune job: {finetune_id}")
    print("\nFirst, let's check our fine-tunes list:")
    
    try:
        finetunes = monitor.list_finetunes()
        print("\nYour fine-tunes:")
        print(json.dumps(finetunes, indent=2))
        
        # Check if the fine-tune ID exists in the list
        if finetune_id not in finetunes.get('finetunes', []):
            print(f"\nNote: Fine-tune ID {finetune_id} not found in your fine-tunes list.")
            print("This is normal for newly started jobs. Continuing to monitor...")
    except Exception as e:
        print(f"Error listing fine-tunes: {e}")
    
    print("\nNow checking job progress...")
    
    while True:
        try:
            result = monitor.check_progress(finetune_id)
            print("\nProgress check result:")
            print(json.dumps(result, indent=2))
            
            # Check the status
            status = result.get('status', '')
            progress = result.get('progress')
            
            if status == 'Ready':
                print("\nFine-tuning is complete!")
                break
            elif status == 'Failed':
                print("\nFine-tuning failed!")
                break
            elif status == 'Error':
                print("\nStatus is Error - job may still be initializing or in queue")
            elif status == 'Pending':
                print("\nJob is pending/in progress...")
                if progress:
                    print(f"Progress: {progress}")
                
            print(f"\nWaiting {check_interval} seconds before next check...")
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError checking progress: {e}")
            print(f"\nWaiting {check_interval} seconds before retrying...")
            time.sleep(check_interval)

if __name__ == "__main__":
    API_KEY = "21006105-1bcc-4969-abab-97e55051d7a3"
    FINETUNE_ID = "92258091-db89-4cf7-97c8-ece585d94231"  # Your latest fine-tune ID
    
    monitor_progress(API_KEY, FINETUNE_ID) 