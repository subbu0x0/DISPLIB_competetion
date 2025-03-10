from ortools.sat.python import cp_model
import json
import os

def solve_displib_problem(problem_json):
    """Solves the train dispatching problem using Constraint Programming."""

    model = cp_model.CpModel()

    trains = problem_json["trains"]
    objective_components = problem_json["objective"]

    train_vars = {}
    resource_vars = {}
    LARGE_INTEGER = 0
    temp = []
    #n = len(trains)
    for i in trains:
        LARGE_INTEGER = 0
        for j in i:
            LARGE_INTEGER += j["min_duration"]
        temp.append(LARGE_INTEGER)
    LARGE_INTEGER = 2*sum(temp)
    #print(temp,n,LARGE_INTEGER)
    #return

    # 1. Create Variables
    for train_index, train_operations in enumerate(trains):
        train_vars[train_index] = {}
        #min_duration_p = 0
        #start_lb_p = 0
        for op_index, op_data in enumerate(train_operations):
            #print(min_duration_train)
            min_duration = op_data.get("min_duration", 0)
            #start_lb = max(op_data.get("start_lb", 0),start_lb_p + min_duration_p) #Successors only
            start_lb = op_data.get("start_lb", 0)
            start_ub = op_data.get("start_ub", LARGE_INTEGER)
            resources = op_data.get("resources", [])

            train_vars[train_index][op_index] = {}
            train_vars[train_index][op_index]["start_time"] = model.NewIntVar(start_lb, start_ub, f"train_{train_index}_op_{op_index}_start")
            train_vars[train_index][op_index]["active"] = model.NewBoolVar(f"active_{train_index}_{op_index}")
            train_vars[train_index][op_index]["end_time"] = model.NewIntVar(start_lb + min_duration, LARGE_INTEGER, f"train_{train_index}_op_{op_index}_end")
            #start_lb_p = start_lb
            #min_duration_p = min_duration
            
            for resource_data in resources:
                resource = resource_data["resource"]
                release_time = resource_data.get("release_time", 0)
                if resource not in resource_vars:
                    resource_vars[resource] = []
                resource_vars[resource].append({
                    "train": train_index,
                    "operation": op_index,
                    "release_time": release_time
                })
        model.Add(train_vars[train_index][0]["active"] == 1)  # Entry operation is always active
    
            

    # 2. Add Successor Constraints (within each train)
    def add_successor_constraints(train_operations, train_index):
        for op_index, op_data in enumerate(train_operations):

            successors = op_data.get("successors", [])
            min_duration = op_data.get("min_duration", 0)

            st_ij = train_vars[train_index][op_index]["start_time"]
            active_ij = train_vars[train_index][op_index]["active"]
            et_ij = train_vars[train_index][op_index]["end_time"]

            model.Add(et_ij - st_ij >= min_duration) # Operation duration is always greater than min_duration

            if not successors: # last operation has no successors
                continue

            successor_vars = [train_vars[train_index][succ_op]["active"] for succ_op in successors]
            model.Add(sum(successor_vars) == 1).OnlyEnforceIf(active_ij)  # Ensure only one successor is chosen

            #successor starts after current operation ends
            for i, succ_op in enumerate(successors):
                chosen = successor_vars[i]
                succ_st = train_vars[train_index][succ_op]["start_time"]

                model.Add( succ_st == et_ij).OnlyEnforceIf([chosen,active_ij])

                resources_needed_for_succ = train_operations[succ_op].get("resources", [])

                c = 0
                # IF all resources needed for the successor are not shared with other operations, then the successor starts immediately after the current operation
                for resource_data in resources_needed_for_succ:
                    resource = resource_data["resource"]
                    if len(resource_vars[resource]) > 1:
                        c = 1
                        break
                if(c == 0):
                    model.Add(et_ij==st_ij + min_duration).OnlyEnforceIf([chosen,active_ij])


    for train_index, train_operations in enumerate(trains):
        add_successor_constraints(train_operations, train_index)


    # 3. Add Resource Constraints (preventing conflicts between trains)
        
    for resource in resource_vars:
        operations_using_resource = resource_vars[resource]

        l = len(operations_using_resource)
        if(l <= 1):
            continue
        for i in range(l):

            op1_data = operations_using_resource[i]
            
            train1_idx = op1_data["train"]
            op1_idx = op1_data["operation"]

            start1 = train_vars[train1_idx][op1_idx]["start_time"]
            end1 = train_vars[train1_idx][op1_idx]["end_time"]
            release_time1 = op1_data["release_time"]
            active1 = train_vars[train1_idx][op1_idx]["active"]



            for j in range(i + 1, l):
                
                op2_data = operations_using_resource[j]

                if train1_idx == op2_data["train"]:
                    continue # Only add constraints for operations from *different* trains

                train2_idx = op2_data["train"]
                op2_idx = op2_data["operation"]

                start2 = train_vars[train2_idx][op2_idx]["start_time"]
                end2 = train_vars[train2_idx][op2_idx]["end_time"]
                release_time2 = op2_data["release_time"]
                active2 = train_vars[train2_idx][op2_idx]["active"]

                # Ensure no overlap: op1 finishes before op2 starts OR op2 finishes before op1 starts
                y = model.NewBoolVar(f"resource_order_{train1_idx}_{op1_idx}_{train2_idx}_{op2_idx}_{resource}")

                model.Add(start1 > end2 + release_time2).OnlyEnforceIf([y, active1, active2])
                model.Add(start2 > end1  + release_time1).OnlyEnforceIf([y.Not(), active1, active2])
        





    # 4. Define Objective Function
    objective_expr = 0
    for obj_component in objective_components:
        if obj_component["type"] == "op_delay":
            train_index = obj_component["train"]
            op_index = obj_component["operation"]
            threshold = obj_component.get("threshold", 0)
            coeff = obj_component.get("coeff", 0)
            increment = obj_component.get("increment", 0)
            delay = model.NewIntVar(0, LARGE_INTEGER, f"delay_train_{train_index}_op_{op_index}")
            model.Add(delay == train_vars[train_index][op_index]["start_time"] - threshold)
            delayed_cost = model.NewBoolVar(f"delayed_cost_{train_index}_{op_index}")
            model.Add(delay > 0 ).OnlyEnforceIf(delayed_cost)
            model.Add(delay <=0 ).OnlyEnforceIf(delayed_cost.Not())
            max_val = model.NewIntVar(0, LARGE_INTEGER, f"max_val_{train_index}_{op_index}")
            model.AddMaxEquality(max_val, [0, delay])
            cost = model.NewIntVar(0, LARGE_INTEGER, f"cost_{train_index}_{op_index}")
            model.Add(cost == coeff * max_val + increment * delayed_cost)
            objective_expr += cost

    if objective_components: # Only Minimize if objective is defined
        model.Minimize(objective_expr)

    #print(model)
    # 5. Call Constraint Solver
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 28800
    status = solver.Solve(model)

    # 6. Process the Solution
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        events = []
        for train_index, train_operations in enumerate(trains):
            for op_index, op_data in enumerate(train_operations):
                start_time = solver.Value(train_vars[train_index][op_index]["start_time"])
                active = solver.Value(train_vars[train_index][op_index]["active"])
                #end_time = solver.Value(train_vars[train_index][op_index]["end_time"])
                if active:
                    events.append({"operation": op_index, "time": start_time, "train": train_index})
        events = sorted(events, key=lambda x: x["time"])

        return {
            "objective_value": solver.ObjectiveValue(),
            "events": events,
        }
    else:
        return None  # No solution found


# Load problem instance (replace with your file path if needed)
file_path = os.path.join(os.path.dirname(__file__), '/Users/subhapravan/Operations Research/DISPLIB/problem_instance.json')
with open(file_path, 'r') as f:
    problem_data = json.load(f)

# Solve it and print result
solution = solve_displib_problem(problem_data)
#output_dict = json.loads(solution)  # Convert string to dict

with open("line1_full_9_solution.json", "w") as json_file:
    json.dump(solution,json_file,indent=4) # Save JSON with formatting

if solution:
    print("Feasible Solution Found:\n", json.dumps(solution, indent=2))
else:
    print("No feasible solution found.")