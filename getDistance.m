function [ distance ] = getDistance( Data, centroid, i, count)

 for j = 1:count
   distance(i, j) = sum((Data(i,:) - centroid(j, :)).^2);
 end

end
    
    
    
    
    
    
    
    
    
