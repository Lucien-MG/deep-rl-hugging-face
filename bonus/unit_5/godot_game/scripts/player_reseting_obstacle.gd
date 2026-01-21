extends Area3D
class_name PlayerResetingObstacle

## An obstacle that resets the player if it enters its area, such as spikes, water.

func _on_body_entered(body):
	var player = body as Player
	if player:
		player.game_over(-1) # Sets reward to -1 as well
