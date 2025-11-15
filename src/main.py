import argparse  # noqa: INP001
import time
from collections import deque

import cv2
import numpy as np

from evio.core.pacer import Pacer
from evio.source.dat_file import BatchRange, DatFileSource
from clustering import SimpleClusterer
from predict_path3D import KalmanFilter

def get_window(
    event_words: np.ndarray,
    time_order: np.ndarray,
    timestamps: np.ndarray,
    win_start: int,
    win_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    This function is from the evio package https://github.com/ahtihelminen/evio.git

    Extract event data from a specific timeframe window.
    This is the KEY FUNCTION to get event stream data from a given timeframe.
    
    Returns:
        x_coords: X coordinates of events
        y_coords: Y coordinates of events
        pixel_polarity: Polarity (ON/OFF) of events
        event_timestamps: Absolute timestamp in microseconds for each event
    """
    # Get indexes corresponding to events within the specified time window
    # time_order contains temporal ordering of events, win_start/win_stop define the timeframe
    event_indexes = time_order[win_start:win_stop]
    
    # Convert event data to 32-bit unsigned integers for bit manipulation
    words = event_words[event_indexes].astype(np.uint32, copy=False)
    
    # Extract the absolute timestamps for all events in this window (in microseconds)
    event_timestamps = timestamps[win_start:win_stop].astype(np.int64, copy=False)
    
    # Extract X coordinates from bits 0-13 (14 bits, max value 16383)
    x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
    
    # Extract Y coordinates from bits 14-27 (14 bits, shifted right by 14)
    y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
    
    # Extract polarity (ON/OFF event) from bits 28-31
    # True = pixel brightness increased, False = pixel brightness decreased
    pixel_polarity = ((words >> 28) & 0xF) > 0

    # Return event data plus the absolute timestamps array
    return x_coords, y_coords, pixel_polarity, event_timestamps

def get_frame(
    window: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    width: int = 1280,
    height: int = 720,
    *,
    base_color: tuple[int, int, int] = (122, 122, 122),  # dark blue
    on_color: tuple[int, int, int] = (255, 255, 255),  # white
    off_color: tuple[int, int, int] = (0, 0, 0),  # black
) -> np.ndarray:
    """
    This function is from the evio package https://github.com/ahtihelminen/evio.git
    Convert event data from a time window into a visual frame for display."""
    # Unpack the event data: coordinates, polarities, and timestamps (timestamps not used for rendering)
    x_coords, y_coords, polarities_on, _ = window
    
    # Create blank frame with base (gray) color
    frame = np.full((height, width, 3), base_color, np.uint8)
    
    # Draw ON events (brightness increase) as white pixels
    frame[y_coords[polarities_on], x_coords[polarities_on]] = on_color
    
    # Draw OFF events (brightness decrease) as black pixels
    frame[y_coords[~polarities_on], x_coords[~polarities_on]] = off_color

    return frame


def draw_hud(
    frame: np.ndarray,
    pacer: Pacer,
    batch_range: BatchRange,
    *,
    color: tuple[int, int, int] = (0, 0, 0),  # black by default
    cluster_count: int = 0,
    drone_detected: bool = False,
    drone_position: tuple[float, float] | None = None,
    estimated_distance: float | None = None,
    rpm_estimation: float | None = None,
) -> None:
    """
    This function is partly from the evio package https://github.com/ahtihelminen/evio.git
    Overlay timing info: wall time, recording time, and playback speed."""
    if pacer._t_start is None or pacer._e_start is None:
        return

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    wall_time_s = time.perf_counter() - pacer._t_start
    rec_time_s = max(0.0, (batch_range.end_ts_us - pacer._e_start) / 1e6)

    if pacer.force_speed:
        first_row_str = (
            f"speed={pacer.speed:.2f}x"
            f"  drops/ms={pacer.instantaneous_drop_rate:.2f}"
            f"  avg(drops/ms)={pacer.average_drop_rate:.2f}"
        )
    else:
        first_row_str = (
            f"(target) speed={pacer.speed:.2f}x  force_speed = False, no drops"
        )

    second_row_str = f"wall={wall_time_s:7.3f}s  rec={rec_time_s:7.3f}s"
    
    # Display current frame time in microseconds
    third_row_str = f"frame_time={batch_range.start_ts_us}us"
    
    # Display clustering statistics
    fourth_row_str = f"clusters={cluster_count}"
    
    # Display drone detection status
    fifth_row_str = f"DRONE: {'Detected' if drone_detected else '-'}"
    
    # Display drone position as ratio of frame dimensions
    if drone_position is not None:
        x_ratio = drone_position[0] / frame_width
        y_ratio = drone_position[1] / frame_height
        sixth_row_str = f"Position: ({x_ratio:.3f}, {y_ratio:.3f})"
    else:
        sixth_row_str = "Position: -"
    
    # Display estimated distance to drone
    if estimated_distance is not None:
        seventh_row_str = f"Distance: {estimated_distance:.2f}m"
    else:
        seventh_row_str = "Distance: -"
    
    # Display RPM
    if rpm_estimation is not None:
        eighth_row_str = f"RPM: {rpm_estimation:.2f}"
    else:
        eighth_row_str = "RPM: -"

    # first row
    cv2.putText(
        frame,
        first_row_str,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    # second row
    cv2.putText(
        frame,
        second_row_str,
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

    # third row
    cv2.putText(
        frame,
        third_row_str,
        (8, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    
    # fourth row
    cv2.putText(
        frame,
        fourth_row_str,
        (8, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    
    # fifth row - drone detection
    cv2.putText(
        frame,
        fifth_row_str,
        (8, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0) if drone_detected else color,
        1,
        cv2.LINE_AA,
    )
    
    # sixth row - drone position
    cv2.putText(
        frame,
        sixth_row_str,
        (8, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    
    # seventh row - estimated distance
    cv2.putText(
        frame,
        seventh_row_str,
        (8, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    
    # eighth row - RPM
    cv2.putText(
        frame,
        eighth_row_str,
        (8, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

def main() -> None:
    """
    This function is from partly the evio package https://github.com/ahtihelminen/evio.git
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("dat", help="Path to .dat file")
    
    """    parser.add_argument(
        "--window", type=float, default=10, help="Windows duration in ms"
    )""" #window duration conflicts with rpm calculations, so removed  
    parser.add_argument(
        "--speed", type=float, default=1, help="Playback speed (1 is real time)"
    )
    parser.add_argument(
        "--force-speed",
        action="store_true",
        help="Force the playback speed by dropping windows",
    )
    args = parser.parse_args()

    # STEP 1: Initialize the event camera data source from .dat file
    # window_length_us defines the timeframe window size in microseconds
    # Here we use 10ms windows 
    src = DatFileSource(
        args.dat, width=1280, height=720, window_length_us=10 * 1000
    )

    # STEP 2: Create a pacer to control playback timing (speed, dropping frames if needed)
    pacer = Pacer(speed=args.speed, force_speed=args.force_speed)

    # STEP 2.5: Initialize simple synchronous clustering
    clusterer = SimpleClusterer(
        flatten_time=True,    # Keep time dimension or not in window
        eps=5.0,             # 10.0 for propellers 0.1ms
        min_samples=20,       # 5 for propellers with 0.1ms
        min_events=100        # Skip clustering if too few events
    )
    
    # Distance estimation calibration parameters
    CALIBRATION_DISTANCE = 30.0  # meters - known distance where drone was measured, note that this is an educated guess
    CALIBRATION_DRONE_SIZE_PIXELS = 80.0  # pixels - drone cluster size at calibration distance


    cv2.namedWindow("Evio Player", cv2.WINDOW_NORMAL)
    
    # STEP 3: Main loop - iterate through time ranges from the event stream
    # src.ranges() generates sequential time windows from the recording
    # pacer.pace() controls the playback speed to match real-time or specified speed

    # Track last clustering time (cluster every 1 second)
    last_cluster_timestamp_us = 0
    latest_centroids = None
    allClusterPoints = None
    
    # Track drone path - array of (x, y) positions
    drone_path = np.empty((0, 2), dtype=np.float64)
    drone_path_max_size = 100  # Prevent unbounded growth
    
    # Track latest distance measurements for averaging (deque for O(1) operations)
    distance_history = deque(maxlen=10)
    
    # Dictionary to store drone-specific Kalman filters
    #The Kalman filter is used in trajectory predictions
    # drone_id -> {'kf': KalmanFilter, 'last_seen': time, 'positions': [...]}
    drones = {}
    next_drone_id = 0
    DRONE_TIMEOUT_S = 2.0  # Remove drone if not seen for 2 seconds
    MAX_TRACE_LENGTH = 100  # Maximum number of points in trace
    rpm = None
    last_rpm_time = 0  # Track last RPM calculation time
    RPM_CALCULATION_INTERVAL_S = 0.5  # Calculate RPM every 0.5 seconds

    for batch_range in pacer.pace(src.ranges()):
        # STEP 4: Extract event data for the current timeframe window
        # batch_range contains start/stop indexes and timestamps for this window
        window = get_window(
            src.event_words,   # Raw event data
            src.order,         # Temporal ordering of events
            src.timestamps,    # Absolute timestamps in microseconds
            batch_range.start, # Start index of this time window
            batch_range.stop,  # Stop index of this time window
        )

        # Get current timestamp (use first event's timestamp from the window)
        current_timestamp_us = window[3][0] if len(window[3]) > 0 else 0
        current_wall_time_s = time.perf_counter()


        # Cluster every 0.05 second to allow enough time for real time tracking
        cluster_frequency_us = 50_000
        if current_timestamp_us - last_cluster_timestamp_us >= cluster_frequency_us:
            # Extract only events within _ms from the start of the window
            CLUSTER_WINDOW_MS = 1
            cluster_window_us = CLUSTER_WINDOW_MS * 1000
            
            # Get timestamps array
            event_timestamps = window[3]
            
            if len(event_timestamps) > 0:
                # Find window start timestamp
                window_start_us = event_timestamps[0]
                
                # Filter events within the cluster window (1ms from start)
                mask = event_timestamps <= (window_start_us + cluster_window_us)
                
                # Extract only the events within the cluster window
                x_coords_filtered = window[0][mask]
                y_coords_filtered = window[1][mask]
                
                # Run clustering on filtered subset
                latest_centroids, allClusterPoints = clusterer.cluster_events(
                    x_coords_filtered,
                    y_coords_filtered
                )
            else:
                latest_centroids = None
                
            last_cluster_timestamp_us = current_timestamp_us
        
        # STEP 5: Convert event data to a visual frame
        frame = get_frame(window)

        # Drone detection based on centroid proximity
        drone_detected = False
        drone_centroids = set()  # Track which centroids are part of drone detection
        drone_position = None  # Current drone position
        drone_relative_size_pixels = None  # Maximum distance between drone centroids
        estimated_distance = None  # Estimated distance to drone in meters
        PROXIMITY_THRESHOLD = 100.0  # Distance threshold in pixels
        matched_drone_id = None
        pos_3D = None

        if latest_centroids and len(latest_centroids) > 1:
            # Filter out None values and convert centroids to numpy array for vectorized operations
            valid_centroids = [c for c in latest_centroids if c is not None]
            
            if len(valid_centroids) > 1:
                centroids_array = np.array(valid_centroids)  
                
                # Ensure it's a 2D array 
                if centroids_array.ndim == 1:
                    centroids_array = centroids_array.reshape(-1, 2)
                
                # Compute pairwise distances using broadcasting 
                diff = centroids_array[:, np.newaxis, :] - centroids_array[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diff**2, axis=2))  # Shape: (n, n)
                
                # Set diagonal to infinity to ignore self-distances
                np.fill_diagonal(distances, np.inf)
                
                # Find centroids that have at least one neighbor within threshold
                has_close_neighbor = np.any(distances <= PROXIMITY_THRESHOLD, axis=1)
                drone_centroids = set(np.where(has_close_neighbor)[0])
                drone_detected = len(drone_centroids) > 0
                
                # Calculate drone relative size and distance estimation
                if drone_detected and len(drone_centroids) >= 2:
                    # Get distances only between drone centroids 
                    drone_indices = np.array(list(drone_centroids))
                    drone_distances = distances[np.ix_(drone_indices, drone_indices)]
                    
                    # Maximum distance between any two drone centroids
                    finite_distances = drone_distances[np.isfinite(drone_distances)]
                    if len(finite_distances) > 0:
                        drone_relative_size_pixels = np.max(finite_distances)
                        
                        # Distance estimation: ratio = distance / calibration_distance
                        # drone_relative_size_pixels / CALIBRATION_DRONE_SIZE_PIXELS = CALIBRATION_DISTANCE / estimated_distance
                        current_distance = (CALIBRATION_DISTANCE * CALIBRATION_DRONE_SIZE_PIXELS) / drone_relative_size_pixels
                        
                        distance_history.append(current_distance)
                        
                        estimated_distance = np.mean(distance_history)
                
                # Find or create drone track
                if drone_centroids:
                    drone_positions = np.array([valid_centroids[i] for i in drone_centroids])
                    avg_position = np.mean(drone_positions, axis=0)
                    pos_3D = np.array([avg_position[0], avg_position[1], 
                                       estimated_distance if estimated_distance is not None else 0.0])
                    drone_position = (float(avg_position[0]), float(avg_position[1]))
                    
                    # keep last 100 positions to prevent unbounded growth
                    if len(drone_path) >= drone_path_max_size:
                        drone_path = drone_path[-drone_path_max_size+1:]
                    drone_path = np.vstack([drone_path, avg_position])
                    
                    best_drone_id = None
                    best_distance = float('inf')
                    ASSOCIATION_THRESHOLD = 150.0  

                    for did, drone_data in drones.items():
                        # Use predicted position if available, else last measured position
                        assoc_pos = None
                        if 'predicted_pos' in drone_data and drone_data['predicted_pos'] is not None:
                            assoc_pos = drone_data['predicted_pos'][:2, 0]
                        elif drone_data.get('last_position') is not None:
                            assoc_pos = drone_data['last_position']
                        
                        if assoc_pos is not None:
                            dist = np.linalg.norm(avg_position - assoc_pos)
                            if dist < best_distance:
                                best_distance = dist
                                best_drone_id = did

                    # create a new drone if no match found within threshold
                    if best_drone_id is not None and best_distance < ASSOCIATION_THRESHOLD:
                        matched_drone_id = best_drone_id
                    else:
                        matched_drone_id = next_drone_id
                        drones[next_drone_id] = {
                            'kf': KalmanFilter(
                                dt=0.05, 
                                sigma_a=3.2,      
                                sigma_pos=3.0,    
                                dims=3,
                                init_pos_var=5.0,  # initial position uncertainty
                                init_vel_var=20.0, # initial velocity uncertainty
                                warmup_updates=2   # shorter warmup period
                            ),
                            'last_position': avg_position,
                            'last_seen': current_wall_time_s,
                            'trajectory_3d': np.empty((0, 3), dtype=np.float64),
                        }
                        next_drone_id += 1

                    if matched_drone_id is not None:
                        drones[matched_drone_id]['last_position'] = avg_position
                        drones[matched_drone_id]['last_seen'] = current_wall_time_s
                        
                        if 'trace' not in drones[matched_drone_id]:
                            drones[matched_drone_id]['trace'] = deque(maxlen=MAX_TRACE_LENGTH)
                        
                        drones[matched_drone_id]['trace'].append(avg_position.copy())
                    
                    # Update drone's Kalman filter
                    if matched_drone_id is not None:
                        drone_kf = drones[matched_drone_id]['kf']
                        
                        # call update
                        drone_kf.update(pos_3D)
                        
                        # Save velocity from the updated state
                        updated_vel = drone_kf.get_velocity()
                        drones[matched_drone_id]['updated_vel'] = np.array(updated_vel, dtype=float)
                    
                    # Generate predicted trajectory 
                    predicted_trajectory = np.empty((0, 3), dtype=np.float64)
                    temp_x = drone_kf.x.copy()
                    temp_P = drone_kf.P.copy()
                    for step in range(20):
                        temp_x = drone_kf.F @ temp_x
                        temp_P = drone_kf.F @ temp_P @ drone_kf.F.T + drone_kf.Q_base * drone_kf.q_scale
                        predicted_trajectory = np.vstack([
                            predicted_trajectory,
                            temp_x[:3, 0]
                        ])
                    drones[matched_drone_id]['predicted_trajectory'] = predicted_trajectory

                    # Predict one step ahead for next frame
                    predicted_pos, predicted_vel = drone_kf.predict()
                    drones[matched_drone_id]['predicted_pos'] = predicted_pos
                    drones[matched_drone_id]['predicted_vel'] = predicted_vel
        
        # Remove stale drones
        stale_drones = [did for did, d in drones.items() 
                       if current_wall_time_s - d['last_seen'] > DRONE_TIMEOUT_S]
        for did in stale_drones:
            del drones[did]
        
        # Draw all active drone predictions and trajectories
        for drone_id, drone_data in drones.items():
            avg_pos = drone_data.get('last_position')
            if avg_pos is not None:
                px, py = int(avg_pos[0]), int(avg_pos[1])

                # Get current estimated distance from Kalman filter
                # Use the first point in predicted_trajectory if available, else fallback to last known
                predicted_trajectory = drone_data.get('predicted_trajectory')
                if predicted_trajectory is not None and len(predicted_trajectory) > 0:
                    z = predicted_trajectory[0][2]
                else:
                    z = drone_data.get('estimated_distance', 30.0)

                # Scale radius: closer = bigger, further = smaller
                # Clamp z to reasonable range to avoid extreme sizes
                min_dist, max_dist = 5.0, 50.0
                z_clamped = np.clip(z, min_dist, max_dist)

                min_radius, max_radius = 8, 32
                radius = int(max_radius - (z_clamped - min_dist) / (max_dist - min_dist) * (max_radius - min_radius))

                # Color: red if close, blue if far
                # Non-linear interpolation for more drastic color change
                close_color = np.array([255, 0, 0], dtype=np.float64)  
                far_color   = np.array([0, 0, 255], dtype=np.float64)   
                t = (z_clamped - min_dist) / (max_dist - min_dist)
                # Apply power function for more drastic transition 
                t_nonlinear = t ** 2
                color_np = (far_color * t_nonlinear + close_color * (1 - t_nonlinear))
                color = (int(color_np[0]), int(color_np[1]), int(color_np[2]))

                # Draw the center tag at the measured centroid mean, scaled and colored
   
                cv2.putText(frame, f"D{drone_id}", (px + radius + 5, py - radius - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                # Draw velocity vector from measured center using CURRENT velocity
                updated_vel = drone_data.get('updated_vel')
                if updated_vel is not None:
                    vel_scale = 2.0
                    vx = float(updated_vel[0])
                    vy = float(updated_vel[1])
                    ex = int(px + vx * vel_scale)
                    ey = int(py + vy * vel_scale)
                    cv2.arrowedLine(frame, (px, py), (ex, ey), color, 2, tipLength=0.1)
                
        # Draw drone path (last 100 positions) as a thin red line
        if len(drone_path) > 1:
            # Get last 100 positions
            path_to_draw = drone_path[-100:]
            # Convert to integer coordinates for drawing
            points = path_to_draw.astype(np.int32)
            # Draw polyline connecting the path points
            cv2.polylines(frame, [points], isClosed=False, color=(0, 0, 255), thickness=1)
        
        # Draw all cluster centroids
        if latest_centroids is not None and len(latest_centroids) > 0:
            for idx, centroid in enumerate(latest_centroids):
                if centroid is not None:
                    cx, cy = centroid
                    # Determine color based on whether this cluster is part of drone detection
                    if idx in drone_centroids:
                        color = (0, 255, 0)  # Green for drone clusters
                    else:
                        color = (0, 0, 255)  # Red for non-drone clusters
                    
                    # Draw circle at centroid position
                    cv2.circle(frame, (int(cx), int(cy)), radius=7, color=color, thickness=2)

        # Find RPM - calculate infrequently (every 0.5 seconds) for performance
        if (latest_centroids is not None and len(latest_centroids) > 0 and 
            allClusterPoints is not None and drone_detected and drone_centroids):
            if current_wall_time_s - last_rpm_time >= RPM_CALCULATION_INTERVAL_S:
                # Use the first drone centroid as RPM center
                rpm_centroid_idx = min(drone_centroids)
                if rpm_centroid_idx < len(latest_centroids) and latest_centroids[rpm_centroid_idx] is not None:
                    rpm_centroid = latest_centroids[rpm_centroid_idx]
                    oscillation_times = rpmDetect(allClusterPoints, rpm_centroid, src, batch_range)
                    if oscillation_times and oscillation_times >= 2000:
                        rpm = 1 / (oscillation_times * 1e-6) * 60
                    last_rpm_time = current_wall_time_s
        
        # STEP 6: Add timing/debugging info overlay to the frame
        draw_hud(
            frame, 
            pacer, 
            batch_range,
            cluster_count=len(latest_centroids) if latest_centroids else 0,
            drone_detected=drone_detected,
            drone_position=drone_position,
            estimated_distance=estimated_distance,
            rpm_estimation=rpm if rpm else None
        )

        # Display the frame
        cv2.imshow("Evio Player", frame)

        # Exit on ESC or 'q' key press
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
            
    cv2.destroyAllWindows()


def rpmDetect(clusterPoints, clusterCentroid, src, batch_range):
    """
    Detect RPM based on oscillation of black values within the smallest circle around the cluster (blade) centroid.
    """

    def next_10_ranges(src, start_ts_us):
        """
        find next 10 ranges after the given start timestamp to convert to windows
        """
        ranges_after_condition = []  # List to store the next 10 ranges
        found_start = False  # Flag to indicate when the condition is met

        for batchRange in src.ranges():
            if not found_start:
                # Check if the current range meets the condition
                if batchRange.start_ts_us >= start_ts_us:
                    found_start = True  # Start collecting ranges after this point
            else:
                # Collect the next 10 ranges
                ranges_after_condition.append(batchRange)
                if len(ranges_after_condition) == 10:
                    break  # Stop after collecting 10 ranges

        return ranges_after_condition

    # Get the pixels within the smallest circle around the cluster centroid
    x_coords_filtered, y_coords_filtered = get_pixels_in_smallest_circle(
        clusterPoints,
        clusterCentroid
    )
    
    ranges_after_condition = next_10_ranges(src, batch_range.start_ts_us)
    

    if len(ranges_after_condition) < 2:
        return None

   
    num_ranges = len(ranges_after_condition)
    blackvalues = np.zeros(num_ranges, dtype=np.int32)
    times = np.zeros(num_ranges, dtype=np.int64)

    
    for i, batch_Range in enumerate(ranges_after_condition):
        # Extract event indexes for the current batch range
        event_indexes = src.order[batch_Range.start:batch_Range.stop]

        # Convert event data to 32-bit unsigned integers for bit manipulation
        words = src.event_words[event_indexes].astype(np.uint32, copy=False)

        # Extract polarities for the filtered pixels
        x_coords = (words & 0x3FFF).astype(np.int32, copy=False)
        y_coords = ((words >> 14) & 0x3FFF).astype(np.int32, copy=False)
        polarities = ((words >> 28) & 0x1).astype(bool)

        # Create a mask to filter events that belong to the smallest circle
        circle_mask = np.isin(x_coords, x_coords_filtered) & np.isin(y_coords, y_coords_filtered)

        # Count false polarities directly without creating intermediate arrays
        false_count = np.sum(~polarities[circle_mask])
        blackvalues[i] = false_count
        times[i] = batch_Range.start_ts_us
        
    
    
    def oscillationPeriod(blackvalues, times):
        """
        Get the time difference between the two highest blackvalues.
        """
        if len(blackvalues) < 2:
            return None

        blackvalues_array = np.asarray(blackvalues)
        times_array = np.asarray(times)
        
        # Get indices of 2 largest values
        top_2_indices = np.argpartition(blackvalues_array, -2)[-2:]
        
        # Calculate time difference
        time_difference = abs(times_array[top_2_indices[0]] - times_array[top_2_indices[1]])

        return time_difference
    
    oscillattion_times = oscillationPeriod(blackvalues, times)
    
    return oscillattion_times

def get_pixels_in_smallest_circle(clusterPoints, clusterCentroid):
    """
    Get all pixel coordinates within the smallest circle that includes all cluster points.
    """
    if not clusterPoints:
        return np.array([]), np.array([])
    
    centroid_x, centroid_y = clusterCentroid

    all_points = np.vstack(clusterPoints)
    cluster_x_coords = all_points[:, 0]
    cluster_y_coords = all_points[:, 1]

    # Calculate the Euclidean distance of each pixel from the centroid
    dx = cluster_x_coords - centroid_x
    dy = cluster_y_coords - centroid_y
    distances_squared = dx * dx + dy * dy
    
    return cluster_x_coords, cluster_y_coords


if __name__ == "__main__":
    main()
