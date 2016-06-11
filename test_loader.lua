require 'data_loader'
require 'hdf5'
local dataLoader = SequentialDB('toydata.h5', 3, 40)
local i_min = dataLoader.batchIndexs:clone() - 1
local i_max = i_min:clone() + dataLoader.rho - 1

for i = 1,dataLoader.bs do
    if i_max[i] > 99 then
        i_min[i] = i_min[i] + 400
        i_max[i] = i_max[i] + 400
    end
end

for i = 1,200 do
    data, labels = dataLoader:getBatch()
    --print (data)
    for i = 1, dataLoader.bs do
        local seq = torch.linspace(i_min[i], i_max[i], dataLoader.rho)
        --if not data[{i}]:eq(seq):all() then
         --   print(dataLoader.batchIndexs)
         --   print(data[{i}])
         --   print(seq)
        --end
        assert(data[{i}]:eq(seq):all())
        --local y = torch.sin(seq)
        --assert((y - labels):sum() < 0.00001)
    end
    i_min:add(1)
    i_max:add(1)
    for i = 1,dataLoader.bs do
        if i_max[i] > 99 and i_min[i] < 99 then
            i_min[i] = 500
            i_max[i] = 539
        elseif i_max[i] > 599 and i_min[i] >= 500 then
            i_min[i] = 0
            i_max[i] = 39
        end
    end

end


