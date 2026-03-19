# 1collect_v5_simultaneous.py
"""
改进版数据采集脚本 - 同步采集版本
✅ 只在驶入环岛的道路上生成车辆
✅ 只生成四轮汽车
✅ 密度配置严格符合HCM 2010 LOS标准
✅ 车辆到达模式：泊松过程 + 负指数分布间隔
✅ 【修改1】Spawn + 数据采集同时进行 180秒
✅ 【修改2】车辆驶出环岛60米外自动销毁

学术依据:
- May (1990): Traffic Flow Fundamentals - 泊松到达过程
"""
import sys
sys.path.append('D:/Carla Simulation')

import carla
import pandas as pd
import numpy as np
import math
import time
from pathlib import Path

# ===== 内联配置 =====
ROUNDABOUT_CENTER = carla.Location(x=0.0, y=0.0, z=0.0)
OUTER_RING_RADIUS = 24.8
COLLECTION_RADIUS = 50.0
DESTROY_RADIUS = 60.0  # 【新增】超过此距离销毁车辆
FRAME_RATE = 10
SCENARIO_DURATION = 180
WARMUP_TIME = 10

# 天气配置
WEATHER_TYPES = ['ClearNoon', 'WetNoon', 'SoftRainNoon', 'HardRainNoon', 'ClearSunset']

WEATHER_SPEED_ADJUSTMENT = {
    'ClearNoon': 0.0, 'WetNoon': 8.0, 'SoftRainNoon': 12.0,
    'HardRainNoon': 20.0, 'ClearSunset': 5.0,
}

# 密度配置 - 严格符合HCM 2010 + 泊松到达
TRAFFIC_DENSITIES = {
    'very_sparse': {
        'target_flow': 300,
        'target_passages': 15,
        'spawn_total': 16,
        'spawn_per_batch': 1,
        'mean_headway': 12.0,
        'min_headway': 3.0,
    },
    'sparse': {
        'target_flow': 500,
        'target_passages': 25,
        'spawn_total': 27,
        'spawn_per_batch': 1,
        'mean_headway': 7.2,
        'min_headway': 2.5,
    },
    'medium': {
        'target_flow': 750,
        'target_passages': 38,
        'spawn_total': 42,
        'spawn_per_batch': 1,
        'mean_headway': 4.8,
        'min_headway': 2.0,
    },
    'dense': {
        'target_flow': 1050,
        'target_passages': 53,
        'spawn_total': 58,
        'spawn_per_batch': 1,
        'mean_headway': 3.4,
        'min_headway': 2.0,
    },
    'very_dense': {
        'target_flow': 1400,
        'target_passages': 70,
        'spawn_total': 75,
        'spawn_per_batch': 1,
        'mean_headway': 2.6,
        'min_headway': 2.0,
    },
}

# 行为配置
BEHAVIOR_SPEED_ADJUSTMENT = {'aggressive': -20.0, 'normal': 0.0, 'cautious': 30.0}
BEHAVIOR_FOLLOWING_DISTANCE = {'aggressive': 1.5, 'normal': 2.5, 'cautious': 4.0}
BEHAVIOR_IGNORE_LIGHTS = {'aggressive': 10, 'normal': 10, 'cautious': 10}

# 数据路径
BASE_DIR = 'D:/Carla Simulation'
RAW_DATA_DIR = f'{BASE_DIR}/data/raw_v5_simultaneous'

TOTAL_SCENARIOS = len(WEATHER_TYPES) * len(TRAFFIC_DENSITIES)

# 要排除的车辆类型
EXCLUDED_VEHICLE_TYPES = [
    'crossbike', 'century', 'omafiets', 'low_rider', 
    'ninja', 'zx125', 'yzf', 'vespa', 'harley', 'kawasaki',
    'yamaha', 'diamondback', 'gazelle', 'bh.crossbike'
]


def get_exponential_headway(mean_headway, min_headway=2.0):
    """生成服从负指数分布的车头时距"""
    headway = np.random.exponential(mean_headway)
    headway = max(headway, min_headway)
    max_headway = mean_headway * 4
    headway = min(headway, max_headway)
    return headway


class SimultaneousCollector:
    """
    改进的数据采集器 - 同步采集版本
    
    核心改进：
    1. Spawn + 数据采集同时进行
    2. 车辆超过60米自动销毁
    3. 整个180秒都在采集数据
    """
    
    def __init__(self):
        print("连接CARLA...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = None
        self.traffic_manager = None
        self.spawned_vehicles = []
        self.vehicle_behaviors = {}
        self.entrance_spawns = {}
        self.four_wheel_bps = []
        
        # 统计信息
        self.total_spawned = 0
        self.total_destroyed = 0
        self.vehicles_entered_core = set()  # 记录进入过核心区的车辆ID
        
        Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    def setup_world(self):
        """配置仿真环境"""
        print("加载Town03...")
        self.world = self.client.load_world('Town03')
        time.sleep(2)
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FRAME_RATE
        self.world.apply_settings(settings)
        
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        
        self._filter_four_wheel_vehicles()
        self._find_entrance_spawn_points()
        
        print("✅ 环境配置完成")
    
    def _filter_four_wheel_vehicles(self):
        """过滤只保留四轮汽车"""
        blueprint_library = self.world.get_blueprint_library()
        all_vehicles = blueprint_library.filter('vehicle.*')
        
        self.four_wheel_bps = []
        excluded_count = 0
        
        for bp in all_vehicles:
            bp_id = bp.id.lower()
            
            is_excluded = any(excluded in bp_id for excluded in EXCLUDED_VEHICLE_TYPES)
            if is_excluded:
                excluded_count += 1
                continue
            
            if bp.has_attribute('number_of_wheels'):
                num_wheels = int(bp.get_attribute('number_of_wheels'))
                if num_wheels != 4:
                    excluded_count += 1
                    continue
            
            self.four_wheel_bps.append(bp)
        
        print(f"  四轮车蓝图: {len(self.four_wheel_bps)}种 (排除{excluded_count}种)")
    
    def _find_entrance_spawn_points(self):
        """找到四个方向的入口spawn点"""
        all_spawns = self.world.get_map().get_spawn_points()
        
        self.entrance_spawns = {
            'north': [], 'south': [], 'east': [], 'west': [],
        }
        
        for sp in all_spawns:
            dist = sp.location.distance(ROUNDABOUT_CENTER)
            
            if not (25 <= dist <= 60):
                continue
            
            dx = sp.location.x - ROUNDABOUT_CENTER.x
            dy = sp.location.y - ROUNDABOUT_CENTER.y
            angle = math.degrees(math.atan2(dy, dx))
            
            to_center = math.degrees(math.atan2(-dy, -dx))
            spawn_yaw = sp.rotation.yaw
            yaw_diff = abs(to_center - spawn_yaw)
            if yaw_diff > 180:
                yaw_diff = 360 - yaw_diff
            
            if yaw_diff > 60:
                continue
            
            spawn_info = {
                'spawn_point': sp,
                'distance': dist,
                'angle': angle,
                'yaw_diff': yaw_diff,
            }
            
            if 45 <= angle <= 135:
                self.entrance_spawns['north'].append(spawn_info)
            elif -135 <= angle <= -45:
                self.entrance_spawns['south'].append(spawn_info)
            elif -45 <= angle <= 45:
                self.entrance_spawns['east'].append(spawn_info)
            else:
                self.entrance_spawns['west'].append(spawn_info)
        
        print(f"\n  入口spawn点统计:")
        total_entrances = 0
        for direction, spawns in self.entrance_spawns.items():
            if spawns:
                spawns.sort(key=lambda x: x['distance'])
                distances = [s['distance'] for s in spawns]
                print(f"    {direction:6s}: {len(spawns):2d}个点, 距离 {min(distances):.1f}-{max(distances):.1f}m")
                total_entrances += len(spawns)
            else:
                print(f"    {direction:6s}: 0个点 ⚠️")
        
        print(f"  总入口点: {total_entrances}个")
    
    def get_entrance_spawn_point(self, direction_idx):
        """获取入口spawn点"""
        directions = ['north', 'south', 'east', 'west']
        
        for i in range(4):
            direction = directions[(direction_idx + i) % 4]
            spawns = self.entrance_spawns.get(direction, [])
            if spawns:
                spawn_info = np.random.choice(spawns)
                return spawn_info['spawn_point'], direction
        
        return None, None
    
    def set_weather(self, weather_name):
        """设置天气"""
        weather_presets = {
            'ClearNoon': carla.WeatherParameters.ClearNoon,
            'WetNoon': carla.WeatherParameters.WetNoon,
            'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
            'HardRainNoon': carla.WeatherParameters.HardRainNoon,
            'ClearSunset': carla.WeatherParameters.ClearSunset,
        }
        self.world.set_weather(weather_presets.get(weather_name, carla.WeatherParameters.ClearNoon))
    
    def set_behavior(self, vehicle, behavior_type, weather_type):
        """设置驾驶行为"""
        behavior_speed = BEHAVIOR_SPEED_ADJUSTMENT.get(behavior_type, 0.0)
        weather_speed = WEATHER_SPEED_ADJUSTMENT.get(weather_type, 0.0)
        final_speed_diff = behavior_speed + weather_speed
        
        self.traffic_manager.vehicle_percentage_speed_difference(vehicle, final_speed_diff)
        
        following_dist = BEHAVIOR_FOLLOWING_DISTANCE.get(behavior_type, 2.5)
        self.traffic_manager.distance_to_leading_vehicle(vehicle, following_dist)
        
        ignore_lights = BEHAVIOR_IGNORE_LIGHTS.get(behavior_type, 10)
        self.traffic_manager.ignore_lights_percentage(vehicle, ignore_lights)
        
        if 'Rain' in weather_type:
            self.traffic_manager.distance_to_leading_vehicle(vehicle, following_dist + 0.5)
    
    def get_random_behavior(self):
        """随机获取行为类型"""
        r = np.random.random()
        if r < 0.25:
            return 'aggressive'
        elif r < 0.75:
            return 'normal'
        else:
            return 'cautious'
    
    def spawn_single_vehicle(self, direction_idx, weather_type):
        """spawn单辆车辆"""
        max_attempts = 20
        
        for attempt in range(max_attempts):
            spawn_point, direction = self.get_entrance_spawn_point(direction_idx + attempt)
            if spawn_point is None:
                continue
            
            bp = np.random.choice(self.four_wheel_bps)
            behavior = self.get_random_behavior()
            
            try:
                vehicle = self.world.spawn_actor(bp, spawn_point)
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                self.set_behavior(vehicle, behavior, weather_type)
                
                self.spawned_vehicles.append(vehicle)
                self.vehicle_behaviors[vehicle.id] = behavior
                self.total_spawned += 1
                
                return vehicle, behavior
                
            except Exception as e:
                for _ in range(3):
                    self.world.tick()
                continue
        
        return None, None
    
    def check_and_destroy_far_vehicles(self):
        """
        【新增】检查并销毁超过60米的车辆
        
        返回: 销毁的车辆数量
        """
        destroyed_count = 0
        vehicles_to_remove = []
        
        for vehicle in self.spawned_vehicles:
            try:
                location = vehicle.get_location()
                dist = location.distance(ROUNDABOUT_CENTER)
                
                # 超过60米就销毁
                if dist > DESTROY_RADIUS:
                    vehicle.destroy()
                    vehicles_to_remove.append(vehicle)
                    destroyed_count += 1
                    self.total_destroyed += 1
                    
            except Exception as e:
                # 车辆可能已经不存在
                vehicles_to_remove.append(vehicle)
        
        # 从列表中移除已销毁的车辆
        for vehicle in vehicles_to_remove:
            if vehicle in self.spawned_vehicles:
                self.spawned_vehicles.remove(vehicle)
            if vehicle.id in self.vehicle_behaviors:
                del self.vehicle_behaviors[vehicle.id]
        
        return destroyed_count
    
    def collect_frame_data(self, frame_id, weather, density):
        """
        采集帧数据
        【修改】同时记录哪些车辆进入了核心区
        """
        data = []
        
        for vehicle in self.spawned_vehicles:
            try:
                transform = vehicle.get_transform()
                velocity = vehicle.get_velocity()
                acceleration = vehicle.get_acceleration()
                
                dx = transform.location.x - ROUNDABOUT_CENTER.x
                dy = transform.location.y - ROUNDABOUT_CENTER.y
                radius = math.sqrt(dx ** 2 + dy ** 2)
                angle = math.atan2(dy, dx)
                
                speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2)
                accel = math.sqrt(acceleration.x ** 2 + acceleration.y ** 2)
                
                behavior = self.vehicle_behaviors.get(vehicle.id, 'unknown')
                
                # 记录进入过核心区的车辆
                if radius <= 25:
                    self.vehicles_entered_core.add(vehicle.id)
                
                data.append({
                    'frame': frame_id,
                    'trackId': vehicle.id,
                    'x': transform.location.x,
                    'y': transform.location.y,
                    'z': transform.location.z,
                    'vx': velocity.x,
                    'vy': velocity.y,
                    'speed': speed,
                    'ax': acceleration.x,
                    'ay': acceleration.y,
                    'accel': accel,
                    'heading': math.radians(transform.rotation.yaw),
                    'radius': radius,
                    'angle': angle,
                    'weather': weather,
                    'traffic_density': density,
                    'behavior_type': behavior
                })
            except:
                continue
        
        return data
    
    def run_scenario(self, scenario_id, weather, density_name, density_config):
        """
        运行单个场景
        【修改】Spawn + 数据采集同时进行
        """
        print(f"\n{'=' * 70}")
        print(f"场景 {scenario_id + 1}/{TOTAL_SCENARIOS}")
        print(f"  天气: {weather}")
        print(f"  密度: {density_name} (目标流量: {density_config['target_flow']} veh/h)")
        print(f"  到达模式: 泊松过程 + 负指数分布间隔")
        print(f"  车辆: 仅四轮汽车")
        print(f"  【新】Spawn+采集同时进行: ✅")
        print(f"  【新】60米外自动销毁: ✅")
        print(f"{'=' * 70}")
        
        self.set_weather(weather)
        self.spawned_vehicles = []
        self.vehicle_behaviors = {}
        self.total_spawned = 0
        self.total_destroyed = 0
        self.vehicles_entered_core = set()
        
        # 预热
        print(f"\n预热 {WARMUP_TIME}秒...")
        for _ in range(WARMUP_TIME * FRAME_RATE):
            self.world.tick()
        
        # ═══════════════════════════════════════════════════════════════════
        # 【核心修改】Spawn + 数据采集同时进行
        # ═══════════════════════════════════════════════════════════════════
        spawn_total = density_config['spawn_total']
        mean_headway = density_config['mean_headway']
        min_headway = density_config['min_headway']
        target_passages = density_config['target_passages']
        target_flow = density_config['target_flow']
        
        print(f"\n⭐ 同步Spawn+采集配置:")
        print(f"  目标流量: {target_flow} veh/h")
        print(f"  平均车头时距: {mean_headway}秒")
        print(f"  总spawn目标: {spawn_total}辆")
        print(f"  销毁距离: {DESTROY_RADIUS}米")
        print(f"  采集时长: {SCENARIO_DURATION}秒")
        print()
        
        all_data = []
        direction_idx = 0
        behavior_counts = {'aggressive': 0, 'normal': 0, 'cautious': 0}
        
        # 下一次spawn的时间（以帧为单位）
        next_spawn_frame = 0
        current_spawn_count = 0
        
        start_time = time.time()
        total_frames = SCENARIO_DURATION * FRAME_RATE
        
        for frame in range(total_frames):
            # 1. 推进仿真
            self.world.tick()
            
            # 2. 检查是否需要spawn新车辆
            if current_spawn_count < spawn_total and frame >= next_spawn_frame:
                vehicle, behavior = self.spawn_single_vehicle(direction_idx, weather)
                
                if vehicle is not None:
                    current_spawn_count += 1
                    behavior_counts[behavior] += 1
                    direction_idx += 1
                    
                    # 打印进度
                    if current_spawn_count % 10 == 0 or current_spawn_count == spawn_total:
                        current_time = frame / FRAME_RATE
                        print(f"  [{current_time:5.1f}s] Spawn: {current_spawn_count}/{spawn_total}, "
                              f"在场车辆: {len(self.spawned_vehicles)}, "
                              f"已销毁: {self.total_destroyed}")
                
                # 计算下一次spawn的时间
                headway = get_exponential_headway(mean_headway, min_headway)
                next_spawn_frame = frame + int(headway * FRAME_RATE)
            
            # 3. 采集数据
            frame_data = self.collect_frame_data(frame, weather, density_name)
            all_data.extend(frame_data)
            
            # 4. 检查并销毁超过60米的车辆
            self.check_and_destroy_far_vehicles()
        
        elapsed = time.time() - start_time
        
        # 清理剩余车辆
        print("\n清理剩余车辆...")
        for vehicle in self.spawned_vehicles:
            try:
                vehicle.destroy()
            except:
                pass
        
        if not all_data:
            print("❌ 未采集到数据")
            return None
        
        df = pd.DataFrame(all_data)
        output_file = Path(RAW_DATA_DIR) / f'scenario_{scenario_id:03d}.csv'
        df.to_csv(output_file, index=False)
        
        # 统计
        unique_tracks = df['trackId'].nunique()
        avg_speed = df['speed'].mean()
        core_tracks = df[df['radius'] <= 25]['trackId'].nunique()
        behavior_dist = df.groupby('behavior_type')['trackId'].nunique()
        
        print(f"\n✅ 完成 ({elapsed:.1f}秒)")
        print(f"   采集数据: {len(df):,}行")
        print(f"   总Spawn: {self.total_spawned}辆")
        print(f"   总销毁: {self.total_destroyed}辆 (超过{DESTROY_RADIUS}米)")
        print(f"   唯一轨迹: {unique_tracks}条")
        print(f"   行为分布: ", end="")
        for behavior, count in behavior_dist.items():
            print(f"{behavior} {count}条, ", end="")
        print()
        print(f"   核心区轨迹: {core_tracks}条 (目标{target_passages}条)")
        print(f"   进入核心区车辆: {len(self.vehicles_entered_core)}辆")
        if unique_tracks > 0:
            print(f"   到达率: {core_tracks/unique_tracks*100:.1f}%")
        print(f"   平均速度: {avg_speed:.2f} m/s ({avg_speed * 3.6:.1f} km/h)")
        
        return df
    
    def cleanup(self):
        """清理资源"""
        print("\n清理环境...")
        
        for vehicle in self.spawned_vehicles:
            try:
                vehicle.destroy()
            except:
                pass
        
        vehicles = self.world.get_actors().filter('vehicle.*')
        for vehicle in vehicles:
            try:
                vehicle.destroy()
            except:
                pass
        
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        
        print("✅ 清理完成")


def main():
    print("=" * 80)
    print("CARLA 环岛数据采集 - 同步采集版本 v5")
    print("=" * 80)
    
    print(f"\n⭐ 核心改进:")
    print(f"  1. 密度配置严格符合HCM 2010 LOS标准")
    print(f"  2. 车辆到达模式：泊松过程 + 负指数分布")
    print(f"  3. 只在入口道路生成四轮汽车")
    print(f"  4. 【新】Spawn + 数据采集同时进行 {SCENARIO_DURATION}秒")
    print(f"  5. 【新】车辆超过 {DESTROY_RADIUS}米 自动销毁")
    
    print(f"\n密度配置:")
    print(f"  {'密度':<12} {'LOS':<4} {'流量':<12} {'车头时距':<10} {'180秒期望'}")
    print(f"  {'-' * 55}")
    
    los_map = {'very_sparse': 'A', 'sparse': 'B', 'medium': 'C', 'dense': 'D', 'very_dense': 'E'}
    for density, config in TRAFFIC_DENSITIES.items():
        los = los_map[density]
        flow = config['target_flow']
        headway = config['mean_headway']
        target = config['target_passages']
        print(f"  {density:<12} {los:<4} {flow:>4} veh/h    {headway:>4.1f}秒      {target:>3}辆")
    
    print(f"\n配置:")
    print(f"  天气类型: {len(WEATHER_TYPES)}种")
    print(f"  密度级别: {len(TRAFFIC_DENSITIES)}种")
    print(f"  总场景数: {TOTAL_SCENARIOS}个")
    print(f"  观测时长: {SCENARIO_DURATION}秒")
    print(f"  销毁距离: {DESTROY_RADIUS}米")
    print(f"  数据输出: {RAW_DATA_DIR}")
    
    confirm = input("\n开始采集？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    collector = SimultaneousCollector()
    collector.setup_world()
    
    scenarios = []
    scenario_id = 0
    for weather in WEATHER_TYPES:
        for density_name, density_config in TRAFFIC_DENSITIES.items():
            scenarios.append({
                'id': scenario_id,
                'weather': weather,
                'density_name': density_name,
                'density_config': density_config,
            })
            scenario_id += 1
    
    successful = 0
    failed = 0
    total_core_tracks = 0
    total_target = 0
    
    start_time = time.time()
    
    for scenario in scenarios:
        try:
            df = collector.run_scenario(
                scenario['id'],
                scenario['weather'],
                scenario['density_name'],
                scenario['density_config']
            )
            if df is not None:
                successful += 1
                core_tracks = df[df['radius'] <= 25]['trackId'].nunique()
                total_core_tracks += core_tracks
                total_target += scenario['density_config']['target_passages']
            else:
                failed += 1
        except Exception as e:
            print(f"❌ 场景 {scenario['id']} 失败: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - start_time
    achievement_rate = total_core_tracks / total_target * 100 if total_target > 0 else 0
    
    print("\n" + "=" * 80)
    print("✅ 采集完成！")
    print("=" * 80)
    print(f"\n统计:")
    print(f"  成功: {successful}/{TOTAL_SCENARIOS} 场景")
    print(f"  失败: {failed}/{TOTAL_SCENARIOS} 场景")
    print(f"  总核心区轨迹: {total_core_tracks}条")
    print(f"  总目标轨迹: {total_target}条")
    print(f"  达成率: {achievement_rate:.1f}%")
    print(f"  实际时长: {total_time / 60:.1f} 分钟")
    print(f"\n数据位置: {RAW_DATA_DIR}")
    
    collector.cleanup()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
