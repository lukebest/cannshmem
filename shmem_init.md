  Team Structure Definition

  `shmem_team_t` (include/host_device/shmem_types.h:71) is a typedef for int, representing a team index.

  The internal team structure `shmemi_team_t` (include/internal/host_device/shmemi_types.h:65-71):
   typedef struct {
       int mype;           // team view, [0, size] - current PE's rank in this team
       int start;          // global view, [0, npes] - starting global PE index
       int stride;         // global view, [1, npes - 1] - stride between PEs
       int size;           // team view - number of PEs in this team
       int team_idx;       // team index in the team pool
   } shmemi_team_t;

  How PEs are Added to Teams

  PEs are not dynamically added to teams. Instead, teams are created by splitting from a parent team using one of
  these methods:

  1. Strided Split (shmem_team_split_strided)
  Located in src/host/team/shmem_team.cpp:242-318

   int32_t shmem_team_split_strided(shmem_team_t parent_team, int32_t pe_start,
                                     int32_t pe_stride, int32_t pe_size,
                                     shmem_team_t *new_team)

   - pe_start: First PE number in the parent team to include
   - pe_stride: Stride between PEs in the parent team
   - pe_size: Total number of PEs in the new team

  The new team's PEs are calculated as:
   - global_pe_start = parent_team.start + pe_start * parent_team.stride
   - global_pe_stride = parent_team.stride * pe_stride
   - global_pe_end = global_pe_start + global_pe_stride * (pe_size - 1)

  2. 2D Cartesian Split (shmem_team_split_2d)
  Located in src/host/team/shmem_team.cpp:321-409

  Splits a parent team into x-axis and y-axis teams based on a 2D Cartesian grid.

  Team Initialization Process

  The initialization process flows through these steps in src/host/init/shmem_init.cpp:

  Step 1: Set Attributes (shmem_set_attr)
   int32_t shmem_set_attr(int32_t my_rank, int32_t n_ranks, uint64_t local_mem_size,
                          const char *ip_port, shmem_init_attr_t **attributes)
   - Sets my_rank (current process rank)
   - Sets n_ranks (total number of ranks/PEs)
   - Sets local_mem_size (shared memory size per rank)
   - Sets ip_port (communication server address)
   - Initializes optional attributes with defaults (MTE engine, 120s timeout)

  Step 2: Initialize with Attributes (shmem_init_attr)
  Located in src/host/init/shmem_init.cpp:746-765

  The flow is:
   1. Check attributes - Validates rank counts and memory size
   2. Version compatibility check
   3. Initialize state from attributes (shmemi_state_init_attr)
      - Sets g_state.mype and g_state.npes
      - Sets heap size: local_mem_size + SHMEM_EXTRA_SIZE
      - Creates default ACL stream
   4. Initialize shared memory heap (shmemi_heap_init)
      - Calls smem_init() and smem_shm_init()
      - Creates shared memory segment via smem_shm_create()
      - Allocates host/device heap base pointers for P2P, SDMA, and RoCE
      - Initializes topology information for each rank
   5. Update device state - Copies host state to device
   6. Initialize memory manager - Sets up heap allocation
   7. Initialize team (shmemi_team_init)
   8. Update device state again - Sync team info to device
   9. Initialize synchronization (shmemi_sync_init)
   10. Control barrier - Synchronizes all ranks

  Step 3: Team Initialization (shmemi_team_init)
  Located in src/host/team/shmem_team.cpp:162-209

   int32_t shmemi_team_init(int32_t rank, int32_t size)

  This function:

   1. Allocates team pool - Creates g_shmem_team_pool array (max 2048 teams)
   2. Initializes SHMEM_TEAM_WORLD (team index 0):
      shmem_team_world.team_idx = SHMEM_TEAM_WORLD;  // 0
      shmem_team_world.start = 0;
      shmem_team_world.stride = 1;
      shmem_team_world.size = size;  // total number of PEs
      shmem_team_world.mype = rank;  // current rank
   3. Allocates device team structure - Copies to device memory
   4. Initializes synchronization pools:
      - shmemi_team_init_sync_pool() - Barrier sync arrays for all teams
      - shmemi_team_init_sync_counter() - Sync counters for all teams
      - shmemi_team_init_core_sync_pool() - Core-level sync (48 AIVs per NPU)
      - shmemi_team_init_core_sync_counter() - Core sync counters
      - shmemi_team_init_partial_barrier_pool() - Partial barrier slots

  Step 4: Creating Sub-teams

  After initialization, sub-teams are created via:

   // Example: Create a team with even-ranked PEs from SHMEM_TEAM_WORLD
   shmem_team_t even_team;
   shmem_team_split_strided(SHMEM_TEAM_WORLD, 0, 2, size/2, &even_team);

   // Example: 2D split
   shmem_team_t x_team, y_team;
   shmem_team_split_2d(parent_team, x_range, &x_team, &y_team);

  Key Constants (include/internal/host_device/shmemi_types.h)

   - SHMEM_MAX_RANKS: 16384 (maximum PEs)
   - SHMEM_MAX_TEAMS: 2048 (maximum teams)
   - SHMEM_TEAM_WORLD: 0 (default team containing all PEs)

✦ The team configuration is immutable once created - you cannot add/remove PEs dynamically. Instead, you create
  new teams by splitting existing ones with specific PE selections.
