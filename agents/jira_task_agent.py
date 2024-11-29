import requests

class JiraTaskAgent:
    def __init__(self, jira_api_url, jira_api_token, user_email):
        self.jira_api_url = jira_api_url
        self.jira_api_token = jira_api_token
        self.user_email = user_email

    def query_jira_api(self, endpoint):
        headers = {
            "Authorization": f"Bearer {self.jira_api_token}",
            "Content-Type": "application/json"
        }
        response = requests.get(f"{self.jira_api_url}/{endpoint}", headers=headers)
        return response.json()

    def get_active_cards(self):
        endpoint = f"search?jql=assignee={self.user_email} AND status=active"
        return self.query_jira_api(endpoint)

    def analyze_and_prioritize_cards(self, cards):
        # Example logic to prioritize cards based on due dates
        sorted_cards = sorted(cards, key=lambda x: x['fields']['duedate'])
        return sorted_cards

    def get_project_context(self, project_key):
        endpoint = f"project/{project_key}"
        return self.query_jira_api(endpoint)

