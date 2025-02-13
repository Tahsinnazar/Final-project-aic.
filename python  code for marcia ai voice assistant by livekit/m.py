import enum
from typing import Annotated
from livekit.agents import llm
import logging

# Logger setup
logger = logging.getLogger("temperature-control")
logger.setLevel(logging.INFO)

# Enum for different zones in the Mars colony
class Zone(enum.Enum):
    CREW_RESIDENTIAL = "crew_residential"
    RESEARCH_LAB = "research_lab"
    MEDICAL_BAY = "medical_bay"
    MARS_GREENHOUSE = "mars_greenhouse"
    AI_SERVER_ROOM = "ai_server_room"
    COMMAND_CENTER = "command_center"
    STORAGE_ROOM = "storage_room"
    MARS_VEHICLE_BAY = "mars_vehicle_bay"
    MARS_VEHICLE_DOCK = "mars_vehicle_dock"

# Crew class to manage colonization team
class Crew:
    def __init__(self):
        # Predefined crew members assigned to different zones
        self.crew_members = {
            "Commander ELITE ARET": Zone.COMMAND_CENTER,
            "Dr. Eman ": Zone.RESEARCH_LAB,
            "Dr. Alishba": Zone.MEDICAL_BAY,
            "Botanist Alisha": Zone.MARS_GREENHOUSE,
            "Engineer Raffay": Zone.AI_SERVER_ROOM,
            "Technician Ali": Zone.STORAGE_ROOM,
            "Pilot Tahsin": Zone.MARS_VEHICLE_BAY,
            "Navigator Tahseen": Zone.MARS_VEHICLE_DOCK,
            "Biologist Amna": Zone.CREW_RESIDENTIAL,
        }

    def assign_crew(self, name: str, zone: Zone):
        """Assign a crew member to a specific zone."""
        self.crew_members[name] = zone
        return f"{name} has been assigned to {zone.value}."

    def get_crew_zone(self, name: str):
        """Get the zone of a specific crew member."""
        if name in self.crew_members:
            return f"{name} is currently in {self.crew_members[name].value}."
        return f"{name} is not assigned to any zone."

    def list_crew(self):
        """List all crew members and their assigned zones."""
        if not self.crew_members:
            return "No crew members assigned yet."
        return "\n".join([f"{name}: {zone.value}" for name, zone in self.crew_members.items()])

# Temperature control system
class AssistantFnc(llm.FunctionContext):
    def __init__(self) -> None:
        super().__init__()

        # Default temperature settings for different zones
        self._temperature = {
            Zone.CREW_RESIDENTIAL: 22,
            Zone.RESEARCH_LAB: 20,
            Zone.MEDICAL_BAY: 24,
            Zone.MARS_GREENHOUSE: 23,
            Zone.AI_SERVER_ROOM: 21,
            Zone.COMMAND_CENTER: 20,
            Zone.STORAGE_ROOM: 19,
            Zone.MARS_VEHICLE_BAY: 18,
            Zone.MARS_VEHICLE_DOCK: 17,
        }

        # Initialize Crew Management
        self.crew = Crew()

    @llm.ai_callable(description="Get the temperature in a specific place.")
    def get_temperature(
        self, zone: Annotated[Zone, llm.TypeInfo(description="The specific zone")]
    ):
        """Returns the current temperature of a specified zone."""
        logger.info("Getting temperature - zone: %s", zone)
        temp = self._temperature[zone]
        return f"The temperature in {zone.value} is {temp}°C."

    @llm.ai_callable(description="Set the temperature in a specific place.")
    def set_temperature(
        self,
        zone: Annotated[Zone, llm.TypeInfo(description="The specific zone")],
        temp: Annotated[int, llm.TypeInfo(description="The temperature to set")],
    ):
        """Sets a new temperature for a specific zone."""
        logger.info("Setting temperature - zone: %s, temp: %s°C", zone, temp)
        self._temperature[zone] = temp
        return f"The temperature in {zone.value} is now set to {temp}°C."

    @llm.ai_callable(description="Assign a crew member to a specific zone.")
    def assign_crew(
        self,
        name: Annotated[str, llm.TypeInfo(description="The crew member's name")],
        zone: Annotated[Zone, llm.TypeInfo(description="The specific zone")],
    ):
        """Assigns a crew member to a designated zone."""
        return self.crew.assign_crew(name, zone)

    @llm.ai_callable(description="Get the assigned zone of a crew member.")
    def get_crew_zone(
        self,
        name: Annotated[str, llm.TypeInfo(description="The crew member's name")],
    ):
        """Retrieves the current location of a crew member."""
        return self.crew.get_crew_zone(name)

    @llm.ai_callable(description="List all crew members and their assigned zones.")
    def list_crew(self):
        """Lists all crew members with their respective zones."""
        return self.crew.list_crew()