import open3d as o3d
import os
import shutil

def process_stls(folder_path):
    # Ensure the directory exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Filter for .stl files
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.stl')]

    if not files:
        print("No STL files found in the directory.")
        return

    print(f"Found {len(files)} STL files.\n")
    print("=" * 60)
    print("MESH INFORMATION")
    print("=" * 60)

    # First pass: Load all meshes and display polygon counts
    mesh_data = []
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        
        # Load the mesh
        mesh = o3d.io.read_triangle_mesh(file_path)
        mesh.compute_vertex_normals()
        
        # Count polygons (triangles)
        poly_count = len(mesh.triangles)
        mesh_data.append({
            'filename': filename,
            'path': file_path,
            'mesh': mesh,
            'poly_count': poly_count
        })
        
        print(f"File: {filename}")
        print(f"  Polygon Count: {poly_count:,}")
        print()

    print("=" * 60)
    
    # Ask if user wants to decimate
    choice = input("\nWould you like to decimate these meshes? (y/n): ").lower()
    
    if choice != 'y':
        print("Operation cancelled.")
        return
    
    try:
        # Get decimation factor
        factor = float(input("Enter target reduction factor (e.g., 0.5 for 50% of original size): "))
        
        if factor <= 0 or factor >= 1:
            print("Factor must be between 0 and 1. Operation cancelled.")
            return
        
        # Create backup folder
        backup_folder = os.path.join(folder_path, "original_meshes_backup")
        os.makedirs(backup_folder, exist_ok=True)
        
        print(f"\nProcessing {len(mesh_data)} meshes...")
        print(f"Original meshes will be backed up to: {backup_folder}\n")
        
        # Process each mesh
        for data in mesh_data:
            filename = data['filename']
            file_path = data['path']
            mesh = data['mesh']
            poly_count = data['poly_count']
            
            target_count = int(poly_count * factor)
            print(f"Processing: {filename}")
            print(f"  Original: {poly_count:,} triangles")
            print(f"  Target:   {target_count:,} triangles")
            
            # Backup original
            backup_path = os.path.join(backup_folder, filename)
            shutil.copy2(file_path, backup_path)
            
            # Apply Quadric Decimation
            decimated_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_count)
            
            final_count = len(decimated_mesh.triangles)
            reduction_pct = ((poly_count - final_count) / poly_count) * 100
            
            # Save with original filename
            o3d.io.write_triangle_mesh(file_path, decimated_mesh)
            
            print(f"  Final:    {final_count:,} triangles ({reduction_pct:.1f}% reduction)")
            print(f"  ✓ Saved as: {filename}\n")
        
        print("=" * 60)
        print("All meshes processed successfully!")
        print(f"Original meshes backed up to: {backup_folder}")
        
    except ValueError:
        print("Invalid input. Operation cancelled.")

if __name__ == "__main__":
    # Change '.' to your specific folder path if needed
    directory = "./" 
    process_stls(directory)