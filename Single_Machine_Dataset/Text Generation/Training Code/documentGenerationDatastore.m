classdef documentGenerationDatastore < matlab.io.Datastore & ...
        matlab.io.datastore.MiniBatchable
    
    properties
        Documents
        Encoding
        MiniBatchSize
    end
    
    properties(SetAccess = protected)
        NumObservations
    end
    
    properties(Access = private)
        CurrentFileIndex
    end
    
    methods
        function ds = documentGenerationDatastore(documents)
            % ds = documentGenerationDatastore(documents) creates a
            % document mini-batch datastore from an array of tokenized
            % documents.
            
            % Add startOfText token to documents
            startTokens = repmat(tokenizedDocument("startOfText"), size(documents));
            documents =  startTokens + documents;
            
            % Set Documents and MiniBatchSize properties.
            ds.Documents = documents;
            ds.MiniBatchSize = 128;
            
            % Create word encoding.
            ds.Encoding = wordEncoding(documents);
            
            % Datastore properties.
            numObservations = numel(documents);
            ds.NumObservations = numObservations;
            ds.CurrentFileIndex = 1;
        end
        
        function tf = hasdata(ds)
            % tf = hasdata(ds) returns true if more data is available.
            
            tf = ds.CurrentFileIndex + ds.MiniBatchSize - 1 ...
                <= ds.NumObservations;
        end
        
        function [data,info] = read(ds)
            % [data,info] = read(ds) read one mini-batch of data.
            
            miniBatchSize = ds.MiniBatchSize;
            enc = ds.Encoding;
            info = struct;
            
            % Read batch of documents.
            startPos = ds.CurrentFileIndex;
            endPos = ds.CurrentFileIndex + miniBatchSize - 1;
            documents = ds.Documents(startPos:endPos);
            
            % Convert documents to sequences.
            paddingValue = 0;
            predictors = doc2sequence(enc,documents, ...
                'PaddingDirection','right', ...
                'PaddingValue',paddingValue);
            
            % Create categorical sequences of responses.
            classNames = [enc.Vocabulary "EndOfText"];
            for i = 1:miniBatchSize
                X = predictors{i};

                % Remove start token.
                X(1) = [];

                % Count padding values.
                nz = sum(X==paddingValue);

                % Convert words to indices and pad with "EndOfText". Append
                % additional "EndOfText" token to account for unpadded
                % sequences.
                words = [ind2word(enc,X(X~=0)) repmat("EndOfText",[1 nz+1])];

                responses{i,1} = categorical(words,classNames);
            end
            
            % Update file index
            ds.CurrentFileIndex = ds.CurrentFileIndex + miniBatchSize;
            
            % Convert data to table.
            data = table(predictors,responses);
        end
        
        function reset(ds)
            % reset(ds) resets the datastore to the start of the data.
            
            ds.CurrentFileIndex = 1;
        end
        
        function dsNew = sort(ds)
            % dsNew = sort(ds) sorts the observations in the datastore by
            % sequence length.
            
            dsNew = copy(ds);
            documents = dsNew.Documents;
            documentLengths = doclength(documents);
            [~,idx] = sort(documentLengths);
            dsNew.Documents = documents(idx);
        end
    end
    
    methods (Hidden = true)
        function frac = progress(ds)
            % frac = progress(ds) returns the percentage of observations
            % read in the datastore.
            
            frac = (ds.CurrentFileIndex - 1) / ds.NumObservations;
        end
    end
end